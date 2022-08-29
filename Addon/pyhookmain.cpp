/*
 * Copyright (C) 2022 Dominik Wojtasik
 * SPDX-License-Identifier: MIT
 */

#include <reshade.hpp>
#include <boost/interprocess/windows_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <sstream>

const int MAX_WIDTH = 3840;
const int MAX_HEIGHT = 2160;

const char* SHMEM_NAME = "PyHookSHMEM_";
const char* EVENT_LOCK_NAME = "PyHookEvLOCK_";
const char* EVENT_UNLOCK_NAME = "PyHookEvUNLOCK_";

extern "C" __declspec(dllexport) const char* NAME = "PyHookAddon";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Passes proccessed buffers to Python pipeline.";

using namespace boost::interprocess;
using namespace reshade::api;

struct SharedData
{
    uint64_t frame_count;
    uint32_t width;
    uint32_t height;
    bool multisampled;
    uint8_t frame[MAX_WIDTH * MAX_HEIGHT * 3];
};

static bool initialized = false;
static HANDLE lock_event;
static HANDLE unlock_event;
static windows_shared_memory shm;
static mapped_region region;
static SharedData* shared_data;
static resource st_resource;

template<typename... Args>
static void reshade_log(Args... inputs)
{
    std::stringstream s;
    (
        [&]{
            s << inputs;
        } (), ...
    );
    reshade::log_message(3, s.str().c_str());
}

static void init_events(DWORD pid)
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

static void init_shmem(DWORD pid)
{
    char shmem_name_with_pid[64];
    sprintf_s(shmem_name_with_pid, "%hs%lu", SHMEM_NAME, pid);
    shm = windows_shared_memory(open_or_create, shmem_name_with_pid, read_write, sizeof(SharedData));
    region = mapped_region(shm, read_write);
    memset(region.get_address(), 0, region.get_size());
    shared_data = reinterpret_cast<SharedData*>(region.get_address());
    reshade_log(shmem_name_with_pid, " initialized @", shared_data);
}

static void init_st_resource(device* const device, resource back_buffer)
{
    // Create staging resource
    const resource_desc desc = device->get_resource_desc(back_buffer);
    const format format = format_to_default_typed(desc.texture.format, 0);
    shared_data->multisampled = desc.texture.samples > 1;
    shared_data->width = desc.texture.width;
    shared_data->height = desc.texture.height;

    device->create_resource(
        resource_desc(shared_data->width, shared_data->height, 1, 1, format, 1, memory_heap::gpu_to_cpu, resource_usage::copy_dest),
        nullptr,
        resource_usage::cpu_access,
        &st_resource
    );

    initialized = true;
}

static bool on_create_swapchain(resource_desc& back_buffer_desc, void* hwnd)
{
    // Automatically disable multisampling whenever possible
    back_buffer_desc.texture.samples = 1;
    return true;
}

static void on_destroy_swapchain(swapchain* swapchain)
{
    if (initialized) {
        device* const device = swapchain->get_device();
        device->destroy_resource(st_resource);
        uint64_t frame_count = shared_data->frame_count;
        memset(shared_data, 0, region.get_size());
        shared_data->frame_count = frame_count;
        initialized = false;
    }
}

static void on_init_swapchain(swapchain* swapchain)
{
    init_st_resource(swapchain->get_device(), swapchain->get_current_back_buffer());
}

static void on_present(command_queue* queue, swapchain* swapchain, const rect*, const rect*, uint32_t, const rect*)
{
    device* const device = swapchain->get_device();
    resource back_buffer = swapchain->get_current_back_buffer();

    if (!initialized) {
        init_st_resource(device, back_buffer);
    }

    // Multisampled buffer cannot be processed
    if (shared_data->multisampled) {
        SetEvent(lock_event);
        WaitForSingleObject(unlock_event, INFINITE);
        return;
    }

    // Copy from back buffer
    command_list* const cmd_list = queue->get_immediate_command_list();
    cmd_list->barrier(back_buffer, resource_usage::present, resource_usage::copy_source);
    cmd_list->barrier(st_resource, resource_usage::cpu_access, resource_usage::copy_dest);
    cmd_list->copy_resource(back_buffer, st_resource);
    cmd_list->barrier(st_resource, resource_usage::copy_dest, resource_usage::cpu_access);
    cmd_list->barrier(back_buffer, resource_usage::copy_source, resource_usage::present);
    queue->flush_immediate_command_list();

    // Map texture to get pixel values
    subresource_data mapped;
    bool result = device->map_texture_region(st_resource, 0, nullptr, map_access::read_only, &mapped);
    for (uint32_t y = 0; y < shared_data->height; ++y)
    {
        for (uint32_t x = 0; x < shared_data->width; ++x)
        {
            // Copy map pixels to frame pixels
            const size_t mpx_index = y * mapped.row_pitch + x * 4;
            const size_t fpx_index = (y * shared_data->width + x) * 3;
            shared_data->frame[fpx_index + 0] = static_cast<const uint8_t*>(mapped.data)[mpx_index + 0];
            shared_data->frame[fpx_index + 1] = static_cast<const uint8_t*>(mapped.data)[mpx_index + 1];
            shared_data->frame[fpx_index + 2] = static_cast<const uint8_t*>(mapped.data)[mpx_index + 2];
        }
    }

    shared_data->frame_count++;

    // Enable Python processiong
    SetEvent(lock_event);
    // Process in Python pipeline
    WaitForSingleObject(unlock_event, INFINITE);
    // Back to ReShade

    // Update texture to set pixel values
    for (uint32_t y = 0; y < shared_data->height; ++y)
    {
        for (uint32_t x = 0; x < shared_data->width; ++x)
        {
            // Copy frame pixels to map pixels
            const size_t mpx_index = y * mapped.row_pitch + x * 4;
            const size_t fpx_index = (y * shared_data->width + x) * 3;
            static_cast<uint8_t*>(mapped.data)[mpx_index + 0] = shared_data->frame[fpx_index + 0];
            static_cast<uint8_t*>(mapped.data)[mpx_index + 1] = shared_data->frame[fpx_index + 1];
            static_cast<uint8_t*>(mapped.data)[mpx_index + 2] = shared_data->frame[fpx_index + 2];
        }
    }
    device->unmap_texture_region(st_resource, 0);

    // Copy to back buffer
    command_list* const new_cmd_list = queue->get_immediate_command_list();
    new_cmd_list->barrier(st_resource, resource_usage::cpu_access, resource_usage::copy_source);
    new_cmd_list->barrier(back_buffer, resource_usage::render_target, resource_usage::copy_dest);
    new_cmd_list->copy_resource(st_resource, back_buffer);
    new_cmd_list->barrier(st_resource, resource_usage::copy_source, resource_usage::cpu_access);
    new_cmd_list->barrier(back_buffer, resource_usage::copy_dest, resource_usage::render_target);
    queue->flush_immediate_command_list();
}

static void register_events()
{
    reshade::register_event<reshade::addon_event::destroy_swapchain>(on_destroy_swapchain);
    reshade::register_event<reshade::addon_event::create_swapchain>(on_create_swapchain);
    reshade::register_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
    reshade::register_event<reshade::addon_event::present>(on_present);
}

static void unregister_events()
{
    reshade::unregister_event<reshade::addon_event::present>(on_present);
    reshade::unregister_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
    reshade::unregister_event<reshade::addon_event::create_swapchain>(on_create_swapchain);
    reshade::unregister_event<reshade::addon_event::destroy_swapchain>(on_destroy_swapchain);
}

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
        break;
    case DLL_PROCESS_DETACH:
        unregister_events();
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}