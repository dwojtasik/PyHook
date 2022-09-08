/*
 * Copyright (C) 2022 Dominik Wojtasik
 * SPDX-License-Identifier: MIT
 *
 * ImGui overlay for PyHook addon
 */

#pragma once

#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#define ImTextureID unsigned long long

#include <imgui.h>
#include <reshade.hpp>
#include <string>
#include <vector>

#include "data_types.h"

struct ComboVar
{
    /// <summary>
    /// The vector of labels for combo items.
    /// </summary>
    std::vector<std::string> items;

    /// <summary>
    /// The actual tooltip for combo variable.
    /// </summary>
    char tooltip[256];
};

/// <summary>
/// ImGui widget for slider with steps.
/// Modifies pvar->value.
/// </summary>
/// <param name="pvar">Pointer to pipeline variable.</param>
/// <param name="is_float">Flag if modified value should be treat as float. Otherwise is treated as integer.</param>
/// <returns>True if value had changed.</returns>
bool SliderWithSteps(PipelineVar* pvar, bool is_float);

/// <summary>
/// ImGui widget for combo box with vector labels support.
/// Modifies pvar->value.
/// </summary>
/// <param name="pvar">Pointer to pipeline variable.</param>
/// <param name="cvar">Pointer to combo box variable.</param>
/// <returns>True if value had changed.</returns>
bool ComboWithVector(PipelineVar* pvar, ComboVar* cvar);

/// <summary>
/// Draws settings overlay in ReShade ImGui addons tab.
/// Renders list of pipelines to be ordered/activated.
/// For each activated pipeline renders it's settings.
/// </summary>
/// <param name="shared_cfg">Pointer to configuration in shared memory.</param>
void DrawSettingsOverlay(SharedConfigData* shared_cfg);