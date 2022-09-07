/*
 * Copyright (C) 2022 Dominik Wojtasik
 * SPDX-License-Identifier: MIT
 */

#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#define ImTextureID unsigned long long

#include <imgui.h>
#include <reshade.hpp>
#include <boost/interprocess/windows_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <map>
#include <sstream>

const int MAX_WIDTH = 3840;
const int MAX_HEIGHT = 2160;
const int MAX_PIPELINES = 100;
const int MAX_PIPELINE_VARS = 10;

const char* SHMEM_NAME = "PyHookSHMEM_";
const char* SHCFG_NAME = "PyHookSHCFG_";
const char* EVENT_LOCK_NAME = "PyHookEvLOCK_";
const char* EVENT_UNLOCK_NAME = "PyHookEvUNLOCK_";

extern "C" __declspec(dllexport) const char* NAME = "PyHook"; //v0.0.1
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Passes proccessed buffers to Python pipeline.";

using namespace std;
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

struct ComboVar
{
    vector<string> items;
    char tooltip[256];
};

struct PipelineVar
{
    bool modified;
    char key[32];
    float value;
    short type;
    float min;
    float max;
    float step;
    char tooltip[256];
};

struct ActivePipelineData
{
    bool enabled;
    bool modified;
    char file[64];
    int var_count;
    PipelineVar settings[MAX_PIPELINE_VARS];
};

struct PipelineData : ActivePipelineData
{
    char name[64];
    char version[12];
    char desc[512];
};

struct ActiveConfigData
{
    bool modified;
};

struct SharedConfigData : ActiveConfigData
{
    int count;
    char order[MAX_PIPELINES][64];
    PipelineData pipelines[MAX_PIPELINES];
};

static bool initialized = false;
static resource st_resource;

static HANDLE lock_event;
static HANDLE unlock_event;

static windows_shared_memory shm;
static mapped_region shm_region;
static SharedData* shared_data;

static windows_shared_memory shc;
static mapped_region shc_region;
static SharedConfigData* shared_cfg;

static map<string, PipelineData*> pipeline_map{};
static map<string, ComboVar> select_box_vars{};
static int selected_pipeline = INT_MAX;
static int hovered_pipeline = INT_MAX;

template<typename... Args>
static void reshade_log(Args... inputs)
{
    stringstream s;
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
        if (st_resource != 0)
            device->destroy_resource(st_resource);
        uint64_t frame_count = shared_data->frame_count;
        memset(shared_data, 0, shm_region.get_size());
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

    shared_data->frame_count++;

    if (!initialized)
        init_st_resource(device, back_buffer);

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
    device->map_texture_region(st_resource, 0, nullptr, map_access::read_only, &mapped);
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

bool SliderWithSteps(PipelineVar* pvar, bool is_float)
{
    char format_float[] = "%.0f";
    if (is_float)
        for (float x = 1.0f; x * pvar->step < 1.0f && format_float[2] < '9'; x *= 10.0f)
            ++format_float[2];

    char value_display[24] = {};
    sprintf_s(value_display, is_float ? format_float : "%0.0f", pvar->value);

    ImGuiStyle& style = ImGui::GetStyle();
    float w = ImGui::CalcItemWidth();
    float spacing = style.ItemInnerSpacing.x;
    float button_sz = ImGui::GetFrameHeight();
    ImGui::PushItemWidth(w - spacing * 2.0f - button_sz * 2.0f);

    ImGui::BeginGroup();

    const int step_count = int((pvar->max - pvar->min) / pvar->step);
    int int_val = int((pvar->value - pvar->min) / pvar->step);
    bool modified = ImGui::SliderInt("##slider", &int_val, 0, step_count, value_display);

    ImGui::PopItemWidth();
    ImGui::SameLine(0, spacing);
    if (ImGui::ArrowButton("<", ImGuiDir_Left))
    {
        if (int_val > 0) {
            int_val--;
            modified = true;
        }
    }
    ImGui::SameLine(0, spacing);
    if (ImGui::ArrowButton(">", ImGuiDir_Right))
    {
        if (int_val < step_count) {
            int_val++;
            modified = true;
        }
    }
    ImGui::SameLine(0, style.ItemInnerSpacing.x);
    ImGui::Text(pvar->key);

    ImGui::EndGroup();

    if (is_float)
        pvar->value = pvar->min + float(int_val) * pvar->step;
    else
        pvar->value = int(pvar->min + float(int_val) * pvar->step);
    return modified;
}

bool ComboWithVector(PipelineVar* pvar, ComboVar* cvar) {
    bool modified = false;
    int int_val = int(pvar->value);

    ImGuiStyle& style = ImGui::GetStyle();
    float w = ImGui::CalcItemWidth();
    float spacing = style.ItemInnerSpacing.x;
    float button_sz = ImGui::GetFrameHeight();
    ImGui::PushItemWidth(w - spacing * 2.0f - button_sz * 2.0f);

    ImGui::BeginGroup();

    if (ImGui::BeginCombo("##combo", cvar->items[int_val].c_str())) {
        for (int i = 0; i < cvar->items.size(); i++) {
            const bool isSelected = (int_val == i);
            if (ImGui::Selectable(cvar->items[i].c_str(), isSelected)) {
                if (int_val != i) {
                    int_val = i;
                    modified = true;
                }
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    ImGui::PopItemWidth();
    ImGui::SameLine(0, spacing);
    if (ImGui::ArrowButton("<", ImGuiDir_Left))
    {
        if (int_val > 0) {
            int_val--;
            modified = true;
        }
    }
    ImGui::SameLine(0, spacing);
    if (ImGui::ArrowButton(">", ImGuiDir_Right))
    {
        if (int_val < cvar->items.size() - 1) {
            int_val++;
            modified = true;
        }
    }
    ImGui::SameLine(0, style.ItemInnerSpacing.x);
    ImGui::Text(pvar->key);

    ImGui::EndGroup();

    if (modified) {
        pvar->value = float(int_val);
    }

    if (ImGui::IsItemHovered() && (strlen(cvar->tooltip) > 0))
        ImGui::SetTooltip(cvar->tooltip);

    return modified;
}

static void draw_overlay(effect_runtime* runtime)
{
    bool modified = false;
    bool display_settings = false;

    ImGui::AlignTextToFramePadding();
    ImGui::BeginChild("##PyHookPipelines", ImVec2(0, 200), true, ImGuiWindowFlags_NoMove);

    if (pipeline_map.size() == 0) {
        for (int idx = 0; idx < shared_cfg->count; idx++) {
            PipelineData* pdata = &shared_cfg->pipelines[idx];
            pipeline_map[pdata->file] = pdata;
        }
    }

    for (int idx = 0; idx < shared_cfg->count; idx++) {
        PipelineData* pdata = pipeline_map[shared_cfg->order[idx]];
        bool pipeline_enabled = pdata->enabled;

        ImGui::PushID(pdata->file);
        ImGui::AlignTextToFramePadding();
        ImGui::BeginGroup();

        const bool draw_border = selected_pipeline == idx;
        if (draw_border)
            ImGui::Separator();

        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[pdata->enabled ? ImGuiCol_Text : ImGuiCol_TextDisabled]);
        stringstream label;
        label << pdata->name;
        if (strlen(pdata->version) > 0)
            label << " " << pdata->version;
        label << " [" << pdata->file << "]";
        if (ImGui::Checkbox(label.str().c_str(), &pipeline_enabled)) {
            ImVec2 move = ImGui::GetMouseDragDelta();
            if (move.x == 0 && move.y == 0) {
                pdata->enabled = pipeline_enabled;
                modified = true;
            }
        }

        ImGui::PopStyleColor();

        if (ImGui::IsItemActive())
            selected_pipeline = idx;
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly))
            hovered_pipeline = idx;
        if (ImGui::IsItemHovered() && !ImGui::IsMouseDragging(ImGuiMouseButton_Left) && (strlen(pdata->version) > 0))
            ImGui::SetTooltip(pdata->desc);

        if (draw_border)
            ImGui::Separator();

        ImGui::EndGroup();
        ImGui::Spacing();
        ImGui::PopID();

        if (pdata->enabled && !display_settings && pdata->var_count > 0)
            display_settings = true;
    }
    ImGui::EndChild();

    if (selected_pipeline < shared_cfg->count && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
    {
        if (hovered_pipeline < shared_cfg->count && hovered_pipeline != selected_pipeline)
        {
            if (hovered_pipeline < selected_pipeline)
                for (int i = selected_pipeline; hovered_pipeline < i; --i)
                    swap(shared_cfg->order[i - 1], shared_cfg->order[i]);
            else
                for (int i = selected_pipeline; i < hovered_pipeline; ++i)
                    swap(shared_cfg->order[i], shared_cfg->order[i + 1]);
            selected_pipeline = hovered_pipeline;
            modified = true;
        }
    }
    else
        selected_pipeline = INT_MAX;

    if (display_settings) {
        ImGui::AlignTextToFramePadding();
        ImGui::BeginChild("##PyHookSettings", ImVec2(0, 200), true, ImGuiWindowFlags_NoMove);

        for (int idx = 0; idx < shared_cfg->count; idx++) {
            PipelineData* pdata = pipeline_map[shared_cfg->order[idx]];
            bool p_modified = false;
            if (pdata->enabled && pdata->var_count > 0) {
                ImGui::AlignTextToFramePadding();
                stringstream label;
                label << pdata->name;
                if (strlen(pdata->version) > 0)
                    label << " " << pdata->version;
                label << " [" << pdata->file << "]";
                if (ImGui::CollapsingHeader(label.str().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                    for (int var_idx = 0; var_idx < pdata->var_count; var_idx++) {
                        PipelineVar* pvar = &pdata->settings[var_idx];
                        bool var_modifed = false;

                        ImGui::PushID(pdata->file, pvar->key);
                        ImGui::AlignTextToFramePadding();

                        if (pvar->type == 0) {
                            bool checked = pvar->value == 1.0f;
                            if (ImGui::Checkbox(pvar->key, &checked)) {
                                pvar->value = checked ? 1.0f : 0.0f;
                                var_modifed = true;
                            }
                        }
                        else if (pvar->type == 1) {
                            if (SliderWithSteps(pvar, false)) {
                                var_modifed = true;
                            }
                        }
                        else if (pvar->type == 3) {
                            stringstream varIdString;
                            varIdString << pdata->file << ":" << pvar->key;
                            string varId = varIdString.str();
                            if (!select_box_vars.count(varId)) {
                                bool are_opts = true;
                                char c;
                                vector<string> opts;
                                stringstream opt;
                                stringstream tooltip_s;
                                // Skip %COMBO[
                                for (int i = 7; i < 256; i++) {
                                    c = pvar->tooltip[i];
                                    if (are_opts) {
                                        if (c == ',' || c == ']') {
                                            opts.push_back(opt.str());
                                            opt.str("");
                                            opt.clear();
                                            if (c == ']')
                                                are_opts = false;
                                        }
                                        else
                                            opt << c;
                                    }
                                    else {
                                        if (c == '\0')
                                            break;
                                        else
                                            tooltip_s << c;
                                    }
                                }
                                ComboVar cvar;
                                cvar.items = opts;
                                sprintf_s(cvar.tooltip, "%s", tooltip_s.str().c_str());
                                select_box_vars[varId] = cvar;
                            }
                            if (ComboWithVector(pvar, &select_box_vars[varId])) {
                                var_modifed = true;
                            }
                        }
                        else {
                            if (SliderWithSteps(pvar, true)) {
                                var_modifed = true;
                            }
                        }

                        if (pvar->type != 3 && ImGui::IsItemHovered() && (strlen(pvar->tooltip) > 0))
                            ImGui::SetTooltip(pvar->tooltip);

                        ImGui::Spacing();
                        ImGui::PopID();

                        if (var_modifed) {
                            pvar->modified = var_modifed;
                            p_modified = true;
                        }
                    }
                }
                ImGui::Separator();
            }
            if (p_modified) {
                pdata->modified = true;
                modified = true;
            }
        }

        ImGui::EndChild();
    }

    if (modified)
        shared_cfg->modified = modified;
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
        reshade::register_overlay(nullptr, draw_overlay);
        break;
    case DLL_PROCESS_DETACH:
        unregister_events();
        reshade::unregister_overlay(nullptr, draw_overlay);
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}