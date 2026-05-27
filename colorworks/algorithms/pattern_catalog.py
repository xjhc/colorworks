from __future__ import annotations

from colorworks.algorithms import registry
from colorworks.domain import (
    PatternKindDef,
    ParameterDef,
    ParameterType,
    OptionDef,
)

# 1. Wave Pattern
registry.register_pattern(PatternKindDef(
    kind="wave",
    name="Wave",
    description="Sinusoidal pattern modulated by density.",
    generation="procedural",
    requires_density=True,
    parameters=[
        ParameterDef(
            "frequency",
            "Frequency (cycles / 100 px)",
            ParameterType.FLOAT,
            default=8.0,
            min=0.5,
            max=64.0,
            step=0.5,
        ),
        ParameterDef(
            "angle_deg",
            "Angle (deg)",
            ParameterType.FLOAT,
            default=45.0,
            min=0.0,
            max=180.0,
            step=1.0,
        ),
        ParameterDef(
            "phase",
            "Phase",
            ParameterType.FLOAT,
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.01,
        ),
    ],
))

# 2. Ordered Dither Pattern
registry.register_pattern(PatternKindDef(
    kind="ordered_dither",
    name="Ordered Dither",
    description="Bayer threshold matrix pattern.",
    generation="procedural",
    requires_density=True,
    parameters=[
        ParameterDef(
            "matrix_size",
            "Bayer Matrix Size",
            ParameterType.INT,
            default=8,
            options=[
                OptionDef(2, "2 x 2"),
                OptionDef(4, "4 x 4"),
                OptionDef(8, "8 x 8"),
                OptionDef(16, "16 x 16"),
            ],
        ),
    ],
))

# 3. Blue Noise Pattern
registry.register_pattern(PatternKindDef(
    kind="blue_noise",
    name="Blue Noise",
    description="Void-and-cluster blue noise threshold mask.",
    generation="procedural",
    requires_density=True,
    parameters=[
        ParameterDef(
            "size",
            "Blue Noise Size",
            ParameterType.INT,
            default=64,
            options=[
                OptionDef(16, "16 x 16"),
                OptionDef(32, "32 x 32"),
                OptionDef(64, "64 x 64"),
                OptionDef(128, "128 x 128"),
            ],
        ),
    ],
))

# 4. Maze Pattern
registry.register_pattern(PatternKindDef(
    kind="maze",
    name="Maze",
    description="Truchet maze pattern generated using diagonal circle arcs.",
    generation="procedural",
    requires_density=True,
    parameters=[
        ParameterDef(
            "scale",
            "Cell Size (px)",
            ParameterType.FLOAT,
            default=16.0,
            min=4.0,
            max=64.0,
            step=1.0,
        ),
        ParameterDef(
            "line_width",
            "Line Width",
            ParameterType.FLOAT,
            default=2.0,
            min=0.5,
            max=10.0,
            step=0.1,
        ),
    ],
))

# 5. Hatch Pattern
registry.register_pattern(PatternKindDef(
    kind="hatch",
    name="Hatch",
    description="Parallel lines pattern modulated by density.",
    generation="procedural",
    requires_density=True,
    requires_orientation=False,
    accepts_orientation=True,
    parameters=[
        ParameterDef(
            "frequency",
            "Frequency (cycles / 100 px)",
            ParameterType.FLOAT,
            default=8.0,
            min=0.5,
            max=64.0,
            step=0.5,
        ),
        ParameterDef(
            "angle_deg",
            "Angle (deg)",
            ParameterType.FLOAT,
            default=45.0,
            min=0.0,
            max=180.0,
            step=1.0,
        ),
        ParameterDef(
            "phase",
            "Phase",
            ParameterType.FLOAT,
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.01,
        ),
    ],
))

# 6. Crosshatch Pattern
registry.register_pattern(PatternKindDef(
    kind="crosshatch",
    name="Crosshatch",
    description="Crosshatch lines pattern modulated by density and orientation.",
    generation="procedural",
    requires_density=True,
    requires_orientation=False,
    accepts_orientation=True,
    parameters=[
        ParameterDef(
            "frequency",
            "Frequency (cycles / 100 px)",
            ParameterType.FLOAT,
            default=8.0,
            min=0.5,
            max=64.0,
            step=0.5,
        ),
        ParameterDef(
            "angle_deg",
            "Angle (deg)",
            ParameterType.FLOAT,
            default=45.0,
            min=0.0,
            max=180.0,
            step=1.0,
        ),
        ParameterDef(
            "phase",
            "Phase",
            ParameterType.FLOAT,
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.01,
        ),
    ],
))
