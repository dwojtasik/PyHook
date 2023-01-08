/*
 * Copyright (C) 2023 Dominik Wojtasik
 * SPDX-License-Identifier: MIT
 *
 * ImGui overlay for PyHook addon
 */

#include <map>
#include <sstream>

#include "imgui_overlay.h"

using namespace std;
using namespace ImGui;

/// <summary>
/// Map of pipeline unique id (filename) to it's pointer in shared memory.
/// </summary>
map<string, PipelineData*> pipeline_map{};

/// <summary>
/// Map of unique combo variable id (pipeline_filename:variable_name) to it's pointer in shared memory.
/// </summary>
map<string, ComboVar> select_box_vars{};

// Index of selected pipeline in settings UI.
int selected_pipeline = INT_MAX;
// Index of the hovered pipeline in settings UI.
int hovered_pipeline = INT_MAX;

bool SliderWithSteps(PipelineVar* pvar, bool is_float)
{
    char format_float[] = "%.0f";
    if (is_float)
        for (float x = 1.0f; x * pvar->step < 1.0f && format_float[2] < '9'; x *= 10.0f)
            ++format_float[2];

    char value_display[24] = {};
    sprintf_s(value_display, is_float ? format_float : "%0.0f", pvar->value);

    ImGuiStyle& style = GetStyle();
    float w = CalcItemWidth();
    float spacing = style.ItemInnerSpacing.x;
    float button_sz = GetFrameHeight();
    PushItemWidth(w - spacing * 2.0f - button_sz * 2.0f);

    BeginGroup();

    const int step_count = int(round((pvar->max - pvar->min) / pvar->step));
    int int_val = int(round((pvar->value - pvar->min) / pvar->step));
    bool modified = SliderInt("##slider", &int_val, 0, step_count, value_display);

    PopItemWidth();
    SameLine(0, spacing);
    if (ArrowButton("<", ImGuiDir_Left))
    {
        if (int_val > 0) {
            int_val--;
            modified = true;
        }
    }
    SameLine(0, spacing);
    if (ArrowButton(">", ImGuiDir_Right))
    {
        if (int_val < step_count) {
            int_val++;
            modified = true;
        }
    }
    SameLine(0, style.ItemInnerSpacing.x);
    Text(pvar->key);

    EndGroup();

    if (is_float)
        pvar->value = pvar->min + int_val * pvar->step;
    else
        pvar->value = int(round(pvar->min + int_val * pvar->step));
    return modified;
}

bool ComboWithVector(PipelineVar* pvar, ComboVar* cvar) {
    bool modified = false;
    int int_val = int(pvar->value);

    ImGuiStyle& style = GetStyle();
    float w = CalcItemWidth();
    float spacing = style.ItemInnerSpacing.x;
    float button_sz = GetFrameHeight();
    PushItemWidth(w - spacing * 2.0f - button_sz * 2.0f);

    BeginGroup();

    if (BeginCombo("##combo", cvar->items[int_val].c_str())) {
        for (int i = 0; i < cvar->items.size(); i++) {
            const bool isSelected = (int_val == i);
            if (Selectable(cvar->items[i].c_str(), isSelected)) {
                if (int_val != i) {
                    int_val = i;
                    modified = true;
                }
            }
            if (isSelected) {
                SetItemDefaultFocus();
            }
        }
        EndCombo();
    }

    PopItemWidth();
    SameLine(0, spacing);
    if (ArrowButton("<", ImGuiDir_Left))
    {
        if (int_val > 0) {
            int_val--;
            modified = true;
        }
    }
    SameLine(0, spacing);
    if (ArrowButton(">", ImGuiDir_Right))
    {
        if (int_val < cvar->items.size() - 1) {
            int_val++;
            modified = true;
        }
    }
    SameLine(0, style.ItemInnerSpacing.x);
    Text(pvar->key);

    EndGroup();

    if (modified) {
        pvar->value = float(int_val);
    }

    if (IsItemHovered() && (strlen(cvar->tooltip) > 0))
        SetTooltip(cvar->tooltip);

    return modified;
}

void DrawSettingsOverlay(SharedConfigData* shared_cfg)
{
    bool modified = false;
    bool display_settings = false;

    AlignTextToFramePadding();
    BeginChild("##PyHookPipelines", ImVec2(0, 250), true, ImGuiWindowFlags_NoMove);

    if (pipeline_map.size() == 0) {
        for (int idx = 0; idx < shared_cfg->count; idx++) {
            PipelineData* pdata = &shared_cfg->pipelines[idx];
            pipeline_map[pdata->file] = pdata;
        }
    }

    for (int idx = 0; idx < shared_cfg->order_count; idx++) {
        PipelineData* pdata = pipeline_map[shared_cfg->order[idx]];
        bool pipeline_enabled = pdata->enabled;

        stringstream id;
        id << pdata->file << ":" << idx;
        PushID(id.str().c_str());
        AlignTextToFramePadding();
        BeginGroup();

        const bool draw_border = selected_pipeline == idx;
        if (draw_border)
            Separator();

        PushStyleColor(ImGuiCol_Text, GetStyle().Colors[pdata->enabled ? ImGuiCol_Text : ImGuiCol_TextDisabled]);
        stringstream label;
        label << pdata->name;
        if (strlen(pdata->version) > 0)
            label << " " << pdata->version;
        label << " [" << pdata->file << "]";
        if (Checkbox(label.str().c_str(), &pipeline_enabled)) {
            ImVec2 move = GetMouseDragDelta();
            if (move.x == 0 && move.y == 0) {
                pdata->enabled = pipeline_enabled;
                modified = true;
            }
        }

        PopStyleColor();

        if (IsItemActive())
            selected_pipeline = idx;
        if (IsItemHovered(ImGuiHoveredFlags_RectOnly))
            hovered_pipeline = idx;
        if (IsItemHovered() && !IsMouseDragging(ImGuiMouseButton_Left) && (strlen(pdata->desc) > 0))
            SetTooltip(pdata->desc);

        if (draw_border)
            Separator();

        EndGroup();
        ImGui::Spacing();
        ImGui::PopID();

        if (pdata->enabled && !display_settings && pdata->var_count > 0)
            display_settings = true;
    }
    EndChild();

    if (selected_pipeline < shared_cfg->order_count && IsMouseDragging(ImGuiMouseButton_Left))
    {
        if (hovered_pipeline < shared_cfg->order_count && hovered_pipeline != selected_pipeline)
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
        AlignTextToFramePadding();
        BeginChild("##PyHookSettings", ImVec2(0, 250), true, ImGuiWindowFlags_NoMove);

        vector<string> displayed{};

        for (int idx = 0; idx < shared_cfg->order_count; idx++) {
            if (count(displayed.begin(), displayed.end(), shared_cfg->order[idx]))
                continue;
            displayed.push_back(shared_cfg->order[idx]);
            PipelineData* pdata = pipeline_map[shared_cfg->order[idx]];
            bool p_modified = false;
            if (pdata->enabled && pdata->var_count > 0) {
                AlignTextToFramePadding();
                stringstream label;
                label << pdata->name;
                if (strlen(pdata->version) > 0)
                    label << " " << pdata->version;
                label << " [" << pdata->file << "]";
                if (CollapsingHeader(label.str().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                    for (int var_idx = 0; var_idx < pdata->var_count; var_idx++) {
                        PipelineVar* pvar = &pdata->settings[var_idx];
                        bool var_modifed = false;

                        PushID(pdata->file, pvar->key);
                        AlignTextToFramePadding();

                        if (pvar->type == 0) {
                            bool checked = pvar->value == 1.0f;
                            if (Checkbox(pvar->key, &checked)) {
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

                        if (pvar->type != 3 && IsItemHovered() && (strlen(pvar->tooltip) > 0))
                            SetTooltip(pvar->tooltip);

                        ImGui::Spacing();
                        ImGui::PopID();

                        if (var_modifed) {
                            pvar->modified = var_modifed;
                            p_modified = true;
                        }
                    }
                }
                Separator();
            }
            if (p_modified) {
                pdata->modified = true;
                modified = true;
            }
        }

        EndChild();
    }

    if (modified)
        shared_cfg->modified = modified;
}

void SetImGuiWindows(ImGuiWindows* windows)
{
    ImGuiContext* context = reinterpret_cast<ImGuiContext*>(reinterpret_cast<uintptr_t>(&ImGui::GetIO()) - offsetof(ImGuiContext, IO));
    windows->active = context->WindowsActiveCount > 1;
    windows->rects.clear();
    for (int i = 0; i < context->Windows.size(); i++) {
        if (context->Windows[i]->Hidden || context->Windows[i]->IsExplicitChild || context->Windows[i]->IsFallbackWindow || context->Windows[i]->BeginCount == 0)
            continue;
        // Skip transparent windows
        if (context->Windows[i]->Flags & 128) {
            // Allow to render original OSD window contents
            if (strcmp(context->Windows[i]->Name, "OSD") == 0)
            {
                float sx = context->Windows[i]->ContentSize.x;
                float sy = context->Windows[i]->ContentSize.y;
                float x = context->Windows[i]->Pos.x + (context->Windows[i]->Size.x - sx) / 2;
                float y = context->Windows[i]->Pos.y + (context->Windows[i]->Size.y - sy) / 2;
                // Calculate text size to skip dummy that forces OSD window width
                PushFont(context->Windows[i]->DrawList->_Data->Font);
                float offset = ImGui::CalcTextSize("1234567890").x; // Consider max OSD text length as 10 chars
                PopFont();
                windows->rects.push_back(ImVec4(x + sx - offset, y, x + sx, y + sy - GetStyle().ItemSpacing.y));
            }
            continue;
        }
        float x = context->Windows[i]->Pos.x;
        float y = context->Windows[i]->Pos.y;
        windows->rects.push_back(ImVec4(x, y, x + context->Windows[i]->Size.x, y + context->Windows[i]->Size.y));
    }
}