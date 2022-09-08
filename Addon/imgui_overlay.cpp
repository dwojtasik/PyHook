/*
 * Copyright (C) 2022 Dominik Wojtasik
 * SPDX-License-Identifier: MIT
 *
 * ImGui overlay for PyHook addon
 */

#include <map>
#include <sstream>

#include "imgui_overlay.h"

using namespace std;
using namespace ImGui;

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

static map<string, PipelineData*> pipeline_map{};
static map<string, ComboVar> select_box_vars{};
static int selected_pipeline = INT_MAX;
static int hovered_pipeline = INT_MAX;

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

    const int step_count = int((pvar->max - pvar->min) / pvar->step);
    int int_val = int((pvar->value - pvar->min) / pvar->step);
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
        pvar->value = pvar->min + float(int_val) * pvar->step;
    else
        pvar->value = int(pvar->min + float(int_val) * pvar->step);
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
    reshade_log(0);
    bool modified = false;
    bool display_settings = false;

    AlignTextToFramePadding();
    BeginChild("##PyHookPipelines", ImVec2(0, 200), true, ImGuiWindowFlags_NoMove);

    reshade_log(1);

    if (pipeline_map.size() == 0) {
        for (int idx = 0; idx < shared_cfg->count; idx++) {
            PipelineData* pdata = &shared_cfg->pipelines[idx];
            pipeline_map[pdata->file] = pdata;
        }
    }

    for (int idx = 0; idx < shared_cfg->count; idx++) {
        PipelineData* pdata = pipeline_map[shared_cfg->order[idx]];
        bool pipeline_enabled = pdata->enabled;

        PushID(pdata->file);
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
        if (IsItemHovered() && !IsMouseDragging(ImGuiMouseButton_Left) && (strlen(pdata->version) > 0))
            SetTooltip(pdata->desc);

        if (draw_border)
            Separator();

        EndGroup();
        Spacing();
        PopID();

        if (pdata->enabled && !display_settings && pdata->var_count > 0)
            display_settings = true;
    }
    EndChild();

    if (selected_pipeline < shared_cfg->count && IsMouseDragging(ImGuiMouseButton_Left))
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
        AlignTextToFramePadding();
        BeginChild("##PyHookSettings", ImVec2(0, 200), true, ImGuiWindowFlags_NoMove);

        for (int idx = 0; idx < shared_cfg->count; idx++) {
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

                        Spacing();
                        PopID();

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