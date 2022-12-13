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
extern "C" __declspec(dllexport) const char* NAME = "PyHook"; //v0.9.0
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Passes processed back buffer to Python pipeline.";

// ReShade version
char* ReShadeVersion = ((char**)GetProcAddress(reshade::internal::get_reshade_module_handle(), "ReShadeVersion"))[0];
// ReShade version with fixed Vulkan/DX12 back buffer index
// Due to: https://github.com/crosire/reshade/commit/d2d9ae4f6704208c74f7b8971c3d66bf01deec28
const char* ReShadeVersionFix = "5.4.3";

// Shared memory for frame data name prefix.
#define SHMEM_NAME "PyHookSHMEM_"
// Shared memory for configuration name prefix.
#define SHCFG_NAME "PyHookSHCFG_"
// Lock event name prefix.
#define EVENT_LOCK_NAME "PyHookEvLOCK_"
// Unlock event name prefix.
#define EVENT_UNLOCK_NAME "PyHookEvUNLOCK_"
// Count of staging resources.
#define RESOURCE_COUNT 3

using namespace std;
using namespace boost::interprocess;
using namespace reshade::api;

// Flag if PyHook app is active.
bool pyhook_active = true;
// Cached PyHook app handle.
HANDLE pyhook_handle = 0;
// Timeout in milliseconds to check if event is in signaled state.
const DWORD timeout_ms = 2000;

// Flag if staging resource was initialized.
bool initialized = false;
// Staging resource to store frame.
resource st_resources[RESOURCE_COUNT];
// Staging resource to upload frame. Only for DX12 and Vulkan.
resource st_up_resources[RESOURCE_COUNT];
// Back buffer format.
format st_format;
// Flag if OpenGL API is used.
bool is_opengl = false;
// Flag if DirectX 12 or Vulkan API is used.
bool is_new_api = false;
// Flag if device is capable of blitting between resources.
bool copy_buffer = false;
// Number of bytes per texture row.
int32_t row_pitch;
// Number of bytes per texture slice.
uint32_t slice_pitch;

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

// Active ImGui windows rects.
ImGuiWindows windows{};

/// <summary>
/// Checks if actual ReShade version has fixed back buffer handle.
/// </summary>
/// <returns>Flag is version has fix.</returns>
bool is_fixed_version()
{
    for (int i = 0; i < strlen(ReShadeVersion); i++) {
        if (ReShadeVersion[i] < ReShadeVersionFix[i])
            return false;
        if (ReShadeVersion[i] > ReShadeVersionFix[i])
            return true;
    }
    return true;
}

// Flag if loaded ReShade version is fixed.
const bool fixed_version = is_fixed_version();

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
/// Destroys PyHook staging resources.
/// </summary>
/// <param name="device">The device pointer.</param>
void destroy_resources(device* device)
{
    for (int i = 0; i < RESOURCE_COUNT; i++) {
        if (st_resources[i] != 0)
            device->destroy_resource(st_resources[i]);
        if (st_up_resources[i] != 0)
            device->destroy_resource(st_up_resources[i]);
    }
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
                destroy_resources(device);
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
    slice_pitch = format_slice_pitch(st_format, row_pitch, desc.texture.height);

    is_opengl = device->get_api() == device_api::opengl;
    is_new_api = device->get_api() == device_api::d3d12 || device->get_api() == device_api::vulkan;
    copy_buffer = device->check_capability(device_caps::copy_buffer_to_texture);

    bool result = true;
    for (int i = 0; i < RESOURCE_COUNT; i++) {
        if (copy_buffer && !is_opengl)
        {
            result = result && device->create_resource(resource_desc(slice_pitch, memory_heap::gpu_to_cpu, resource_usage::copy_dest), nullptr, resource_usage::copy_dest, &st_resources[i]);
            if (is_new_api) {
                result = result && device->create_resource(resource_desc(slice_pitch, memory_heap::cpu_to_gpu, resource_usage::cpu_access), nullptr, resource_usage::cpu_access, &st_up_resources[i]);
            }
        }
        else
        {
            result = result && device->create_resource(resource_desc(desc.texture.width, desc.texture.height, 1, 1, st_format, 1, memory_heap::gpu_to_cpu, resource_usage::copy_dest), nullptr, resource_usage::copy_dest, &st_resources[i]);
            if (is_new_api) {
                result = result && device->create_resource(resource_desc(desc.texture.width, desc.texture.height, 1, 1, st_format, 1, memory_heap::cpu_to_gpu, resource_usage::cpu_access), nullptr, resource_usage::cpu_access, &st_up_resources[i]);
            }
        }
        if (!result) {
            break;
        }
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
/// If old not fixed version is used windows for ImGui will be skipped in mapping.
/// </summary>
/// <param name="mapped">Mapped texture.</param>
/// <param name="previous">Pointer to previous texture used to read ImGui UI pixels.</param>
void write_rgb(subresource_data mapped, subresource_data* previous = 0)
{
    bool skip_imgui = !fixed_version && is_new_api && windows.active;
    auto mapped_data = static_cast<uint8_t*>(mapped.data);
    auto previous_data = skip_imgui && previous != 0 ? static_cast<uint8_t*>(previous->data) : 0;
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
                if (skip_imgui && windows.HasPixel(x, y))
                {
                    if (previous != 0)
                        *reinterpret_cast<uint32_t*>(mapped_data + mpx_index) = *reinterpret_cast<uint32_t*>(previous_data + mpx_index);
                    continue;
                }
                auto a = 3 << 30;
                auto b = (shared_data->frame[fpx_index + b_idx] * 4) << 20;
                auto g = (shared_data->frame[fpx_index + g_idx] * 4) << 10;
                auto r = shared_data->frame[fpx_index + r_idx] * 4;
                *reinterpret_cast<uint32_t*>(mapped_data + mpx_index) = a | b | g | r;
            }
            continue;
        }
        for (uint32_t x = 0; x < shared_data->width; ++x)
        {
            const uint32_t mpx_index = y * row_pitch + x * bytes;
            const uint32_t fpx_index = (y * shared_data->width + x) * 3;
            if (skip_imgui && windows.HasPixel(x, y))
            {
                if (previous != 0)
                {
                    mapped_data[mpx_index + r_idx] = previous_data[mpx_index + r_idx];
                    if (g_idx != 3)
                        mapped_data[mpx_index + g_idx] = previous_data[mpx_index + g_idx];
                    if (b_idx != 3)
                        mapped_data[mpx_index + b_idx] = previous_data[mpx_index + b_idx];
                }
                continue;
            }
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
        destroy_resources(swapchain->get_device());
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
/// Process frame function.
/// Read RGB values from ReShade back buffer and store them in shared memory.
/// Wait for Python processing.
/// Read RGB values from shared memory and write them into ReShade back buffer.
/// </summary>
/// <param name="device">The device pointer.</param>
/// <param name="back_buffer">The current back buffer.</param>
/// <param name="queue">The command queue.</param>
static void process_frame(device* const device, resource back_buffer, command_queue* queue) {
    // Used staging resource.
    const int st_idx = shared_data->frame_count % RESOURCE_COUNT;

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
    if (copy_buffer && !is_opengl)
    {
        cmd_list->barrier(back_buffer, resource_usage::present, resource_usage::copy_source);
        cmd_list->copy_texture_to_buffer(back_buffer, 0, nullptr, st_resources[st_idx], 0, 0, 0);
        cmd_list->barrier(back_buffer, resource_usage::copy_source, resource_usage::present);
    }
    else
    {
        cmd_list->barrier(back_buffer, resource_usage::present, resource_usage::copy_source);
        cmd_list->copy_texture_region(back_buffer, 0, nullptr, st_resources[st_idx], 0, nullptr);
        cmd_list->barrier(back_buffer, resource_usage::copy_source, resource_usage::present);
    }
    queue->flush_immediate_command_list();
    queue->wait_idle();

    // Map texture to get pixel values
    subresource_data mapped = {};
    bool map_result;
    if (copy_buffer && !is_opengl)
    {
        map_result = device->map_buffer_region(st_resources[st_idx], 0, UINT64_MAX, map_access::read_write, &mapped.data);
        mapped.row_pitch = row_pitch;
    }
    else
        map_result = device->map_texture_region(st_resources[st_idx], 0, nullptr, is_opengl ? map_access::read_write : map_access::read_only, &mapped);

    if (!map_result) {
        reshade_log("Staging texture cannot be mapped. Detaching...");
        destroy_resources(device);
        pyhook_active = false;
        return;
    }

    read_rgb(mapped);

    // Enable Python processing
    SetEvent(lock_event);
    // Process in Python pipeline
    if (wait_for_data(device))
        return;
    // Back to ReShade

    // Write new pixel values
    if (is_new_api) {
        subresource_data mapped_up = {};
        bool map_up_result;
        if (copy_buffer && !is_opengl)
        {
            map_up_result = device->map_buffer_region(st_up_resources[st_idx], 0, UINT64_MAX, map_access::write_only, &mapped_up.data);
            mapped_up.row_pitch = row_pitch;
            mapped_up.slice_pitch = slice_pitch;
        }
        else
            map_up_result = device->map_texture_region(st_up_resources[st_idx], 0, nullptr, map_access::write_only, &mapped_up);

        if (!map_up_result) {
            reshade_log("Upload staging texture cannot be mapped. Detaching...");
            destroy_resources(device);
            pyhook_active = false;
            return;
        }

        write_rgb(mapped_up, &mapped);
    }
    else {
        write_rgb(mapped);
    }

    if (copy_buffer && !is_opengl) {
        device->unmap_buffer_region(st_resources[st_idx]);
        if (is_new_api)
            device->unmap_buffer_region(st_up_resources[st_idx]);
    }
    else {
        device->unmap_texture_region(st_resources[st_idx], 0);
        if (is_new_api)
            device->unmap_texture_region(st_up_resources[st_idx], 0);
    }

    // Wait for all resources mapping in DirectX 12
    if (shared_data->frame_count < RESOURCE_COUNT && device->get_api() == device_api::d3d12)
        return;

    // Copy to back buffer
    if (!is_new_api)
        cmd_list->barrier(st_resources[st_idx], resource_usage::copy_dest, resource_usage::copy_source);
    cmd_list->barrier(back_buffer, resource_usage::present, resource_usage::copy_dest);
    if (copy_buffer && !is_opengl)
        cmd_list->copy_buffer_to_texture(is_new_api ? st_up_resources[st_idx] : st_resources[st_idx], 0, 0, 0, back_buffer, 0);
    else
        cmd_list->copy_resource(is_new_api ? st_up_resources[st_idx] : st_resources[st_idx], back_buffer);
    if (!is_new_api)
        cmd_list->barrier(st_resources[st_idx], resource_usage::copy_source, resource_usage::copy_dest);
    cmd_list->barrier(back_buffer, resource_usage::copy_dest, resource_usage::present);
    queue->flush_immediate_command_list();
    queue->wait_idle();
}

/// <summary>
/// ReShade addon callback.
/// See reshade::addon_event::present
/// </summary>
static void on_present(command_queue* queue, swapchain* swapchain, const rect*, const rect*, uint32_t, const rect*)
{
    if (!pyhook_active || (!fixed_version && is_new_api))
        return;

    process_frame(swapchain->get_device(), swapchain->get_current_back_buffer(), queue);
}

/// <summary>
/// ReShade addon callback.
/// Used as fallback for older, not fixed ReShade version where
/// on_present gives invalid handle for back buffer.
/// See reshade::addon_event::on_reshade_present
/// </summary>
static void on_reshade_present(effect_runtime* runtime)
{
    // Fallback only for d3d12 and vulkan
    if (!pyhook_active || fixed_version || !is_new_api)
        return;

    process_frame(runtime->get_device(), runtime->get_current_back_buffer(), runtime->get_command_queue());
}

/// <summary>
/// ReShade addon callback.
/// Used as fallback for older, not fixed ReShade version where
/// on_present gives invalid handle for back buffer.
/// See reshade::addon_event::on_reshade_overlay
/// </summary>
static void on_reshade_overlay(effect_runtime*)
{
    // Fallback only for d3d12 and vulkan
    if (!pyhook_active || fixed_version || !is_new_api)
        return;

    SetImGuiWindows(&windows);
}

/// <summary>
/// ReShade addon ImGui overlay callback.
/// See reshade::register_overlay
/// </summary>
static void draw_overlay(effect_runtime*)
{
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
    reshade::register_event<reshade::addon_event::reshade_overlay>(on_reshade_overlay);
    reshade::register_event<reshade::addon_event::reshade_present>(on_reshade_present);
}

/// <summary>
/// Unregisters ReShade addon callbacks.
/// </summary>
void unregister_events()
{
    reshade::unregister_event<reshade::addon_event::reshade_present>(on_reshade_present);
    reshade::unregister_event<reshade::addon_event::reshade_overlay>(on_reshade_overlay);
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