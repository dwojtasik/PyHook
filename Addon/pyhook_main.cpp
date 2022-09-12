/*
 * Copyright (C) 2022 Dominik Wojtasik
 * SPDX-License-Identifier: MIT
 *
 * PyHook addon
 */

#include "imgui_overlay.h"

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/windows_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <sstream>

 // Export special variables for ReShade addon.
extern "C" __declspec(dllexport) const char* NAME = "PyHook"; //v0.0.1
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Passes proccessed back buffer to Python pipeline.";

// Shared memory for frame data name prefix.
#define SHMEM_NAME "PyHookSHMEM_"
// Shared memory for configuration name prefix.
#define SHCFG_NAME "PyHookSHCFG_"
// Lock event name prefix.
#define EVENT_LOCK_NAME "PyHookEvLOCK_"
// Unlock event name prefix.
#define EVENT_UNLOCK_NAME "PyHookEvUNLOCK_"

using namespace std;
using namespace boost::interprocess;
using namespace reshade::api;

// Flag if PyHook app is active.
bool pyhook_active = true;
// Cached PyHook app handle.
HANDLE pyhook_handle = 0;
// Timeout in millis to check if event is in signaled state.
const DWORD timeout_ms = 2000;

// Flag if staging resource was initialized.
bool initialized = false;
// Staging resource to store frame.
resource st_resource;
// Back buffer format.
format st_format;
// Flag if DirectX API is used.
bool is_directx = true;
// Flag if device is capable of blitting between resources.
bool copy_buffer = false;
// Number of bytes per texture row.
int32_t row_pitch;

// Lock event to handle signals.
HANDLE lock_event;
// Unlock event to handle signals.
HANDLE unlock_event;

// Shared memory for frame.
windows_shared_memory shm;
// Shared memory (frame) region.
mapped_region shm_region;
// Pointer to shared frame data
SharedData* shared_data;

// Shared memory for configuration.
windows_shared_memory shc;
// Shared memory (configuration) region.
mapped_region shc_region;
// Pointer to shared configuration data.
SharedConfigData* shared_cfg;

/// <summary>
/// Logs message to ReShade log.
/// </summary>
/// <typeparam name="...Args">Varargs template.</typeparam>
/// <param name="...inputs">Objects to log.</param>
template<typename... Args>
void reshade_log(Args... inputs)
{
    stringstream s;
    (
        [&]{
            s << inputs;
        } (), ...
    );
    reshade::log_message(3, s.str().c_str());
}

/// <summary>
/// Initializes events required for synchronization.
/// Creates lock and unlock events.
/// </summary>
/// <param name="pid">Owner process ID.</param>
void init_events(DWORD pid)
{
    wchar_t event_lock_name_with_pid[64];
    swprintf_s(event_lock_name_with_pid, L"%hs%lu", EVENT_LOCK_NAME, pid);
    lock_event = CreateEventW(NULL, FALSE, FALSE, event_lock_name_with_pid);
    reshade_log(EVENT_LOCK_NAME, pid, " initialized @", lock_event);

    wchar_t event_unlock_name_with_pid[64];
    swprintf_s(event_unlock_name_with_pid, L"%hs%lu", EVENT_UNLOCK_NAME, pid);
    unlock_event = CreateEventW(NULL, FALSE, FALSE, event_unlock_name_with_pid);
    reshade_log(EVENT_UNLOCK_NAME, pid, " initialized @", unlock_event);
}

/// <summary>
/// Initializes shared memory regions.
/// Creates shared memory for frame and configuration.
/// </summary>
/// <param name="pid">Owner process ID.</param>
void init_shmem(DWORD pid)
{
    char shmem_name_with_pid[64];
    sprintf_s(shmem_name_with_pid, "%hs%lu", SHMEM_NAME, pid);
    shm = windows_shared_memory(open_or_create, shmem_name_with_pid, read_write, sizeof(SharedData));
    shm_region = mapped_region(shm, read_write);
    memset(shm_region.get_address(), 0, shm_region.get_size());
    shared_data = reinterpret_cast<SharedData*>(shm_region.get_address());
    reshade_log(shmem_name_with_pid, " initialized @", shared_data);

    char shcfg_name_with_pid[64];
    sprintf_s(shcfg_name_with_pid, "%hs%lu", SHCFG_NAME, pid);
    shc = windows_shared_memory(open_or_create, shcfg_name_with_pid, read_write, sizeof(SharedConfigData));
    shc_region = mapped_region(shc, read_write);
    memset(shc_region.get_address(), 0, shc_region.get_size());
    shared_cfg = reinterpret_cast<SharedConfigData*>(shc_region.get_address());
    reshade_log(shcfg_name_with_pid, " initialized @", shared_cfg);
}


/// <summary>
/// Waits for data from Python processing.
/// If PyHook app does not exits anymore will clean up and detach addon DLL.
/// </summary>
/// <param name="device">The device pointer to clean up resources.</param>
/// <returns>True if processing should be stopped.</returns>
bool wait_for_data(device* device)
{
    while (true) {
        DWORD wait_result = WaitForSingleObject(unlock_event, timeout_ms);
        if (wait_result == WAIT_OBJECT_0) {
            break;
        }
        DWORD status;
        if (GetExitCodeProcess(pyhook_handle, &status)) {
            if (status != STILL_ACTIVE) {
                reshade_log("Connected PyHook app does not exists anymore. Detaching...");
                if (st_resource != 0)
                    device->destroy_resource(st_resource);
                pyhook_active = false;
                return true;
            }
        }
    }
    return false;
}

/// <summary>
/// Initializes staging resource for frame processing.
/// </summary>
/// <param name="device">ReShade device interface pointer.</param>
/// <param name="back_buffer">ReShade back buffer resource.</param>
/// <returns>True if staging resource was created.</returns>
bool init_st_resource(device* const device, resource back_buffer)
{
    const resource_desc desc = device->get_resource_desc(back_buffer);
    shared_data->multisampled = desc.texture.samples > 1;
    shared_data->width = desc.texture.width;
    shared_data->height = desc.texture.height;

    st_format = format_to_default_typed(desc.texture.format, 0);
    row_pitch = format_row_pitch(st_format, desc.texture.width);
    if (device->get_api() == device_api::d3d12)
        row_pitch = (row_pitch + 255) & ~255;
    const uint32_t slice_pitch = format_slice_pitch(st_format, row_pitch, desc.texture.height);

    is_directx = device->get_api() != device_api::opengl && device->get_api() != device_api::vulkan;
    copy_buffer = device->check_capability(device_caps::copy_buffer_to_texture);

    bool result;
    if (copy_buffer && is_directx)
    {
        result = device->create_resource(resource_desc(slice_pitch, memory_heap::gpu_to_cpu, resource_usage::copy_dest), nullptr, resource_usage::copy_dest, &st_resource);
        device->set_resource_name(st_resource, "PyHook staging buffer");
    }
    else
    {
        result = device->create_resource(resource_desc(desc.texture.width, desc.texture.height, 1, 1, st_format, 1, memory_heap::gpu_to_cpu, resource_usage::copy_dest), nullptr, resource_usage::copy_dest, &st_resource);
        device->set_resource_name(st_resource, "PyHook staging texture");
    }

    initialized = true;
    return result;
}

/// <summary>
/// Reads RGB component values from mapped texture.
/// </summary>
/// <param name="mapped">Mapped texture.</param>
void read_rgb(subresource_data mapped)
{
    auto mapped_data = static_cast<const uint8_t*>(mapped.data);
    for (uint32_t y = 0; y < shared_data->height; ++y)
    {
        short r_idx = 0, g_idx = 1, b_idx = 2; //_idx = 3 means that component is set to 0.
        short bytes = 4; // For RGBA channels.
        switch (st_format) {
        case format::r8_unorm:
            bytes = 1;
            g_idx = 3;
            b_idx = 3;
            break;
        case format::r8g8_unorm:
            bytes = 2;
            b_idx = 3;
            break;
        case format::b8g8r8a8_unorm:
        case format::b8g8r8x8_unorm:
            r_idx = 2;
            b_idx = 0;
            break;
        case format::b10g10r10a2_unorm:
            r_idx = 2;
            b_idx = 0;
        case format::r10g10b10a2_unorm:
            for (uint32_t x = 0; x < shared_data->width; ++x)
            {
                const uint32_t mpx_index = y * row_pitch + x * bytes;
                const uint32_t fpx_index = (y * shared_data->width + x) * 3;
                const uint32_t color = *reinterpret_cast<const uint32_t*>(mapped_data + mpx_index);
                shared_data->frame[fpx_index + r_idx] = ((color & 0x000003FF) / 4) & 0xFF;
                shared_data->frame[fpx_index + g_idx] = (((color & 0x000FFC00) >> 10) / 4) & 0xFF;
                shared_data->frame[fpx_index + b_idx] = (((color & 0x3FF00000) >> 20) / 4) & 0xFF;
            }
            continue;
        }
        for (uint32_t x = 0; x < shared_data->width; ++x)
        {
            const uint32_t mpx_index = y * row_pitch + x * bytes;
            const uint32_t fpx_index = (y * shared_data->width + x) * 3;
            shared_data->frame[fpx_index + 0] = mapped_data[mpx_index + r_idx];
            shared_data->frame[fpx_index + 1] = b_idx == 3 ? 0 : mapped_data[mpx_index + g_idx];
            shared_data->frame[fpx_index + 2] = g_idx == 3 ? 0 : mapped_data[mpx_index + b_idx];
        }
    }
}

/// <summary>
/// Write RGB component values to mapped texture.
/// </summary>
/// <param name="mapped">Mapped texture.</param>
void write_rgb(subresource_data mapped)
{
    auto mapped_data = static_cast<uint8_t*>(mapped.data);
    for (uint32_t y = 0; y < shared_data->height; ++y)
    {
        short r_idx = 0, g_idx = 1, b_idx = 2; //_idx = 3 means that component is set to 0.
        short bytes = 4; // For RGBA channels.
        switch (st_format) {
        case format::r8_unorm:
            bytes = 1;
            g_idx = 3;
            b_idx = 3;
            break;
        case format::r8g8_unorm:
            bytes = 2;
            b_idx = 3;
            break;
        case format::b8g8r8a8_unorm:
        case format::b8g8r8x8_unorm:
            r_idx = 2;
            b_idx = 0;
            break;
        case format::b10g10r10a2_unorm:
            r_idx = 2;
            b_idx = 0;
        case format::r10g10b10a2_unorm:
            for (uint32_t x = 0; x < shared_data->width; ++x)
            {
                const uint32_t mpx_index = y * row_pitch + x * bytes;
                const uint32_t fpx_index = (y * shared_data->width + x) * 3;
                uint32_t color = 3 << 30; // Max alpha channel value.
                color += shared_data->frame[fpx_index + r_idx] * 4;
                color += (shared_data->frame[fpx_index + g_idx] * 4) << 10;
                color += (shared_data->frame[fpx_index + b_idx] * 4) << 20;
                mapped_data[mpx_index] = color;
            }
            continue;
        }
        for (uint32_t x = 0; x < shared_data->width; ++x)
        {
            const uint32_t mpx_index = y * row_pitch + x * bytes;
            const uint32_t fpx_index = (y * shared_data->width + x) * 3;
            mapped_data[mpx_index + r_idx] = shared_data->frame[fpx_index + 0];
            if (g_idx != 3)
                mapped_data[mpx_index + g_idx] = shared_data->frame[fpx_index + 1];
            if (b_idx != 3)
                mapped_data[mpx_index + b_idx] = shared_data->frame[fpx_index + 2];
        }
    }
}

/// <summary>
/// ReShade addon callback.
/// See reshade::addon_event::create_swapchain
/// </summary>
static bool on_create_swapchain(resource_desc& back_buffer_desc, void* hwnd)
{
    if (!pyhook_active)
        return false;
    // Automatically disable multisampling whenever possible
    back_buffer_desc.texture.samples = 1;
    return true;
}

/// <summary>
/// ReShade addon callback.
/// See reshade::addon_event::destroy_swapchain
/// </summary>
static void on_destroy_swapchain(swapchain* swapchain)
{
    if (!pyhook_active)
        return;
    if (initialized) {
        device* const device = swapchain->get_device();
        if (st_resource != 0)
            device->destroy_resource(st_resource);
        uint64_t frame_count = shared_data->frame_count;
        memset(shared_data, 0, shm_region.get_size());
        shared_data->frame_count = frame_count;
        initialized = false;
    }
}

/// <summary>
/// ReShade addon callback.
/// See reshade::addon_event::init_swapchain
/// </summary>
static void on_init_swapchain(swapchain* swapchain)
{
    if (!pyhook_active)
        return;

    if (!init_st_resource(swapchain->get_device(), swapchain->get_current_back_buffer()))
    {
        reshade_log("Staging resource cannot be created. Detaching...");
        pyhook_active = false;
        return;
    }
}

/// <summary>
/// ReShade addon callback.
/// See reshade::addon_event::present
/// </summary>
static void on_present(command_queue* queue, swapchain* swapchain, const rect*, const rect*, uint32_t, const rect*)
{
    if (!pyhook_active)
        return;

    device* const device = swapchain->get_device();
    resource back_buffer = swapchain->get_current_back_buffer();

    if (device->get_api() == device_api::d3d12 || device->get_api() == device_api::vulkan)
    {
        reshade_log(device->get_api() == device_api::d3d12 ? "DirectX 12" : "Vulkan", " is not supported. Detaching...");
        if (st_resource != 0)
            device->destroy_resource(st_resource);
        pyhook_active = false;
        return;
    }

    shared_data->frame_count++;

    if (!initialized)
    {
        if (!init_st_resource(device, back_buffer))
        {
            reshade_log("Staging resource cannot be created. Detaching...");
            pyhook_active = false;
            return;
        }
    }

    if (pyhook_handle == 0) {
        if (shared_cfg->pyhook_pid == 0)
            return;
        pyhook_handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, shared_cfg->pyhook_pid);
    }

    // Multisampled buffer cannot be processed
    if (shared_data->multisampled) {
        SetEvent(lock_event);
        wait_for_data(device);
        return;
    }

    // Copy from back buffer
    command_list* const cmd_list = queue->get_immediate_command_list();
    if (copy_buffer && is_directx)
    {
        cmd_list->barrier(back_buffer, resource_usage::present, resource_usage::copy_source);
        cmd_list->copy_texture_to_buffer(back_buffer, 0, nullptr, st_resource, 0, shared_data->width, shared_data->height);
    }
    else
    {
        cmd_list->barrier(back_buffer, resource_usage::present, resource_usage::copy_source);
        cmd_list->copy_texture_region(back_buffer, 0, nullptr, st_resource, 0, nullptr);
    }

    queue->wait_idle();

    // Map texture to get pixel values
    subresource_data mapped = {};
    bool map_result;
    if (copy_buffer && is_directx)
    {
        map_result = device->map_buffer_region(st_resource, 0, UINT64_MAX, map_access::read_only, &mapped.data);
        mapped.row_pitch = row_pitch;
    }
    else
        map_result = device->map_texture_region(st_resource, 0, nullptr, is_directx ? map_access::read_only : map_access::read_write, &mapped);

    if (!map_result) {
        reshade_log("Staging texture cannot be mapped. Detaching...");
        if (st_resource != 0)
            device->destroy_resource(st_resource);
        pyhook_active = false;
        return;
    }

    read_rgb(mapped);

    // Enable Python processiong
    SetEvent(lock_event);
    // Process in Python pipeline
    if (wait_for_data(device))
        return;
    // Back to ReShade

    write_rgb(mapped);

    if (copy_buffer && is_directx)
        device->unmap_buffer_region(st_resource);
    else
        device->unmap_texture_region(st_resource, 0);

    // Copy to back buffer
    cmd_list->barrier(st_resource, resource_usage::copy_dest, resource_usage::copy_source);
    cmd_list->barrier(back_buffer, resource_usage::copy_source, resource_usage::copy_dest);
    cmd_list->copy_resource(st_resource, back_buffer);
    cmd_list->barrier(st_resource, resource_usage::copy_source, resource_usage::copy_dest);
    cmd_list->barrier(back_buffer, resource_usage::copy_dest, resource_usage::present);
    queue->wait_idle();
}

/// <summary>
/// ReShade addon ImGui overlay callback.
/// See reshade::register_overlay
/// </summary>
static void draw_overlay(effect_runtime*) {
    if (!pyhook_active)
        return;
    DrawSettingsOverlay(shared_cfg);
}

/// <summary>
/// Registers ReShade addon callbacks.
/// </summary>
void register_events()
{
    reshade::register_event<reshade::addon_event::destroy_swapchain>(on_destroy_swapchain);
    reshade::register_event<reshade::addon_event::create_swapchain>(on_create_swapchain);
    reshade::register_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
    reshade::register_event<reshade::addon_event::present>(on_present);
}

/// <summary>
/// Unregisters ReShade addon callbacks.
/// </summary>
void unregister_events()
{
    reshade::unregister_event<reshade::addon_event::present>(on_present);
    reshade::unregister_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
    reshade::unregister_event<reshade::addon_event::create_swapchain>(on_create_swapchain);
    reshade::unregister_event<reshade::addon_event::destroy_swapchain>(on_destroy_swapchain);
}

/// <summary>
/// Unloads addon DLL.
/// </summary>
/// <param name="hModule">DLL handle.</param>
void unload_self(HMODULE hModule)
{
    while (pyhook_active)
        Sleep(500);
    shared_memory_object::remove(shm.get_name());
    shared_memory_object::remove(shc.get_name());
    // Clear lock signal before unload.
    WaitForSingleObject(lock_event, 0);
    FreeLibraryAndExitThread(hModule, NULL);
}

/// <summary>
/// DLL entrypoint.
/// </summary>
/// <param name="hModule">Handle to DLL module.</param>
/// <param name="fdwReason">Reason for calling function.</param>
/// <param name="">Reserved.</param>
/// <returns></returns>
BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID)
{
    const DWORD pid = GetCurrentProcessId();
    switch (fdwReason)
    {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(hModule)) {
            return FALSE;
        }
        init_events(pid);
        init_shmem(pid);
        register_events();
        reshade::register_overlay(nullptr, draw_overlay);
        DisableThreadLibraryCalls(hModule);
        CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)unload_self, hModule, NULL, NULL);
        break;
    case DLL_PROCESS_DETACH:
        unregister_events();
        reshade::unregister_overlay(nullptr, draw_overlay);
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}