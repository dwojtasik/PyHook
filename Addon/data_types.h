/*
 * Copyright (C) 2022 Dominik Wojtasik
 * SPDX-License-Identifier: MIT
 *
 * Data types for PyHook addon
 */

#pragma once

#include <reshade.hpp>

// Const values for shared memory allocation.
//// Max resolution width.
#define MAX_WIDTH 3840
//// Max resolution height.
#define MAX_HEIGHT 2160
//// Max pipeline definitions.
#define MAX_PIPELINES 100
//// Max variable count per pipeline.
#define MAX_PIPELINE_VARS 10

struct SharedData
{
    /// <summary>
    /// Holds actual frame count started on addon registration.
    /// </summary>
    uint64_t frame_count;

    /// <summary>
    /// Actual frame width.
    /// </summary>
    uint32_t width;

    /// <summary>
    /// Actual frame height.
    /// </summary>
    uint32_t height;

    /// <summary>
    /// Flag if back buffer texture was multisampled.
    /// If true, Python processing will be skipped.
    /// </summary>
    bool multisampled;

    /// <summary>
    /// Frame array with pixel data.
    /// Each pixel is enocded as R,G,B component
    /// Array structure: [R,G,B,R,G,B,...] from upper left screen corner, row by row.
    /// </summary>
    uint8_t frame[MAX_WIDTH * MAX_HEIGHT * 3];
};

struct PipelineVar
{
    /// <summary>
    /// Flag if given pipeline variable was modified.
    /// </summary>
    bool modified;

    /// <summary>
    /// The name of pipeline variable.
    /// </summary>
    char key[32];

    /// <summary>
    /// The actual value of pipeline variable.
    /// </summary>
    float value;

    /// <summary>
    /// Type of variable, where:
    /// 0 - bool
    /// 1 - int
    /// 2 - float
    /// 3 - int for combo box display
    /// </summary>
    short type;

    /// <summary>
    /// Minimal value allowed.
    /// </summary>
    float min;

    /// <summary>
    /// Maximal value allowed.
    /// </summary>
    float max;

    /// <summary>
    /// Change step between min/max values.
    /// </summary>
    float step;

    /// <summary>
    /// The variable tooltip to be displayed.
    /// Can contain unique syntax for combo box, as %COMBO[x,y,z]Tooltip.
    /// </summary>
    char tooltip[256];
};

struct ActivePipelineData
{
    /// <summary>
    /// Flag if pipeline is enabled for processing.
    /// </summary>
    bool enabled;

    /// <summary>
    /// Flag if pipeline had it variables modified.
    /// </summary>
    bool modified;

    /// <summary>
    /// Unique pipeline identifier - it's file.
    /// </summary>
    char file[64];

    /// <summary>
    /// Number of variables that pipeline has.
    /// </summary>
    int var_count;

    /// <summary>
    /// Pipeline settings.
    /// </summary>
    PipelineVar settings[MAX_PIPELINE_VARS];
};

struct PipelineData : ActivePipelineData
{
    /// <summary>
    /// Pipeline display name.
    /// </summary>
    char name[64];

    /// <summary>
    /// Pipeline version.
    /// </summary>
    char version[12];

    /// <summary>
    /// Pipeline description.
    /// </summary>
    char desc[512];
};

struct ActiveConfigData
{
    /// <summary>
    /// Flag if configuration was changed and should be
    /// processed by Python runtime.
    /// </summary>
    bool modified;
};

struct SharedConfigData : ActiveConfigData
{
    /// <summary>
    /// The actual count of loaded pipelines.
    /// </summary>
    int count;

    /// <summary>
    /// Order of the pipeliens to be processed.
    /// </summary>
    char order[MAX_PIPELINES][64];

    /// <summary>
    /// Loaded pipelines.
    /// </summary>
    PipelineData pipelines[MAX_PIPELINES];
};