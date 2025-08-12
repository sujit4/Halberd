"""
Page Navigation URL : app/automator
Page Description : Manage (add/edit/schedule/import/export) and execute playbooks.
"""

import json
import os
import threading
from datetime import date

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import (
    ALL,
    MATCH,
    callback,
    callback_context,
    dcc,
    html,
    no_update,
    register_page,
)
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from attack_techniques.technique_registry import TechniqueRegistry
from core.Constants import AUTOMATOR_OUTPUT_DIR, AUTOMATOR_PLAYBOOKS_DIR
from core.Functions import (
    AddNewSchedule,
    GetAllPlaybooks,
    generate_technique_info,
    get_playbook_stats,
    parse_execution_report,
    playbook_viz_generator,
)
from core.playbook.playbook import Playbook
from core.playbook.playbook_error import PlaybookError
from core.playbook.playbook_step import PlaybookStep

# Register page to app
register_page(__name__, path="/automator", name="Automator")


def create_playbook_manager_layout():
    """Creates the playbook management interface layout"""
    return html.Div(
        [
            dbc.Row(
                [
                    # Left Panel - Playbook List
                    dbc.Col(
                        [
                            # Primary management buttons
                            dbc.Row(
                                [
                                    dbc.Col(
                                        # New playbook button
                                        dbc.Button(
                                            [
                                                DashIconify(
                                                    icon="mdi:plus",
                                                    width=24,
                                                    height=24,
                                                    className="me-2",
                                                ),
                                                "New Playbook",
                                            ],
                                            id="open-creator-win-playbook-button",
                                            n_clicks=0,
                                            className="me-2 halberd-button-secondary",
                                            style={"width": "100%"},
                                        ),
                                        md=4,
                                    ),
                                    dbc.Col(
                                        # Import playbook button
                                        dcc.Upload(
                                            id="upload-playbook",
                                            children=dbc.Button(
                                                [
                                                    DashIconify(
                                                        icon="material-symbols:upload-file",  # Upload file icon for import
                                                        width=24,
                                                        height=24,
                                                        className="me-2",
                                                    ),
                                                    "Import",
                                                ],
                                                id="import-pb-button",
                                                n_clicks=0,
                                                className="me-2 halberd-button-secondary",
                                                style={"width": "100%"},
                                            ),
                                        ),
                                        md=4,
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            # View progress button
                                            dbc.Button(
                                                [
                                                    DashIconify(
                                                        icon="mdi:progress-clock",
                                                        width=20,
                                                        className="me-2",
                                                    ),
                                                    "View Progress",
                                                ],
                                                id="view-progress-button",
                                                n_clicks=0,
                                                className="me-2 halberd-button-secondary",
                                                style={"width": "100%"},
                                            ),
                                            id="view-progress-button-container",
                                        ),
                                        md=4,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            # Search bar
                            html.Div(
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(
                                                DashIconify(
                                                    icon="mdi:magnify",
                                                    width=24,
                                                    height=24,
                                                    className="text-muted",
                                                ),
                                                className="bg-halberd-dark",
                                            ),
                                            dbc.Input(
                                                id="playbook-search",
                                                placeholder="Search Playbook...",
                                                type="text",
                                                className="bg-halberd-dark halberd-text halberd-input",
                                            ),
                                        ],
                                        className="w-100",
                                    )
                                ],
                                className="pb-3",
                            ),
                            # Playbook list
                            html.Div(
                                id="playbook-list-container",
                                style={"overflowY": "auto", "height": "76vh"},
                            ),
                        ],
                        width=4,
                        className="bg-halberd-dark",
                    ),
                    # Right Panel - Playbook Visualization
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                DashIconify(
                                                    icon="mdi:information-outline",  # Information icon
                                                    width=48,
                                                    height=48,
                                                    className="text-muted mb-3 me-3",
                                                ),
                                            ),
                                            dbc.Row(
                                                html.P(
                                                    "Select a playbook to view details"
                                                )  # Default message when no playbook is selected
                                            ),
                                        ],
                                        className="halberd-text text-muted",
                                        style={
                                            "textAlign": "center",
                                            "height": "50vh",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                        },
                                    )
                                ],
                                id="playbook-visualization-container",
                                className="d-flex justify-content-center align-items-center ms-4 p-1",
                            )
                        ],
                        width=8,
                        className="p-0",
                    ),
                ],
                className="g-0 flex-fill",
            ),
            # Status Bar
            html.Div(
                [
                    # Left side - Ready status
                    html.Div(
                        [
                            html.Div(
                                "Ready",
                                className="bg-success rounded-circle me-2",
                                style={"width": "8px", "height": "8px"},
                            ),
                        ],
                        className="d-flex align-items-center text-muted",
                    ),
                    # Right side - Stats
                    html.Div(id="playbook-stats", className="text-muted"),
                ],
                className="d-flex justify-content-between p-2 border-top border-secondary",
            ),
            # Add stores and intervals for progress tracking
            dcc.Interval(
                id="execution-interval",
                interval=1000,  # 1 second refresh
                disabled=True,
            ),
            # Element to trigger download/export of playbooks
            dcc.Download(id="download-pb-config-file"),
            # Memory store to save selected playbook context
            dcc.Store(id="selected-playbook-data", data={}),
            dcc.Store(id="selected-playbook-data-editor-memory-store", data={}),
            # Primary off canvas to support various workflows
            dbc.Offcanvas(
                id="automator-offcanvas",
                is_open=False,
                placement="end",
                style={
                    "width": "50%",  # Set width to 50% of screen
                    "max-width": "none",  # Override default max-width
                },
                className="bg-halberd-dark halberd-offcanvas halberd-text",
            ),
            # Off canvas for playbook editing workflow
            generate_playbook_editor_offcanvas(),
            # Add progress off-canvas
            create_execution_progress_offcanvas(),
        ],
        className="bg-halberd-dark d-flex flex-column",
        style={"minHeight": "91vh", "padding-right": "20px", "padding-left": "20px"},
    )


def create_playbook_item(playbook_config):
    """
    Creates a playbook item with click selection functionality and action buttons.
    Makes the entire card clickable while maintaining separate button actions.

    Args:
        playbook_config: Playbook configuration object containing playbook metadata

    Returns:
        dash.html.Div: A clickable playbook card component with actions
    """
    return html.Div(
        [  # Wrapper div for click handling
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            # Content section
                            dbc.Row(
                                [
                                    # Main content column
                                    dbc.Col(
                                        [
                                            # Title and metadata section
                                            html.Div(
                                                [
                                                    # Title
                                                    DashIconify(
                                                        icon="mdi:file-document-outline",
                                                        width=22,
                                                        height=22,
                                                        className="text-muted me-1 mb-2",
                                                    ),
                                                    html.Span(
                                                        playbook_config.name,
                                                        className="mb-2 halberd-brand text-xl",
                                                    ),
                                                    # Metadata row
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                [
                                                                    DashIconify(
                                                                        icon="mdi:account",
                                                                        width=18,
                                                                        className="me-1 mb-2 text-muted",
                                                                    ),
                                                                    html.Span(
                                                                        playbook_config.author,
                                                                        className="text-muted halberd-text me-3",
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Span(
                                                                [
                                                                    DashIconify(
                                                                        icon="mdi:calendar",
                                                                        width=18,
                                                                        className="me-1 mb-2 text-muted",
                                                                    ),
                                                                    html.Span(
                                                                        playbook_config.creation_date,
                                                                        className="text-muted halberd-text",
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                ]
                                            ),
                                            # Description div with fixed height
                                            html.Div(
                                                html.P(
                                                    playbook_config.description[:100]
                                                    + "..."
                                                    if len(playbook_config.description)
                                                    > 100
                                                    else playbook_config.description,
                                                    className="mb-0 text-muted lh-base halberd-typography",
                                                ),
                                                style={
                                                    "minHeight": "45px",
                                                    "maxHeight": "45px",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                        ],
                                        width=9,
                                    ),
                                    # Actions column with fixed width
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    # Primary Action
                                                    dbc.Button(
                                                        [
                                                            DashIconify(
                                                                icon="mdi:play",
                                                                width=16,
                                                                className="me-2",
                                                            ),
                                                            "Execute",
                                                        ],
                                                        id={
                                                            "type": "execute-playbook-button",
                                                            "index": playbook_config.yaml_file,
                                                        },
                                                        size="sm",
                                                        className="w-100 mb-2 halberd-button",
                                                    ),
                                                    # Secondary Actions
                                                    dbc.ButtonGroup(
                                                        [
                                                            dbc.Button(
                                                                DashIconify(
                                                                    icon="mdi:pencil",
                                                                    width=16,
                                                                ),
                                                                id={
                                                                    "type": "edit-playbook-button",
                                                                    "index": playbook_config.yaml_file,
                                                                },
                                                                color="light",
                                                                size="sm",
                                                                title="Edit",
                                                                className="px-2",
                                                            ),
                                                            dbc.Button(
                                                                DashIconify(
                                                                    icon="mdi:calendar",
                                                                    width=16,
                                                                ),
                                                                id={
                                                                    "type": "open-schedule-win-playbook-button",
                                                                    "index": playbook_config.yaml_file,
                                                                },
                                                                color="light",
                                                                size="sm",
                                                                title="Schedule",
                                                                className="px-2",
                                                            ),
                                                            dbc.Button(
                                                                DashIconify(
                                                                    icon="mdi:download",
                                                                    width=16,
                                                                ),
                                                                id={
                                                                    "type": "open-export-win-playbook-button",
                                                                    "index": playbook_config.yaml_file,
                                                                },
                                                                color="light",
                                                                size="sm",
                                                                title="Export",
                                                                className="px-2",
                                                            ),
                                                            dbc.Button(
                                                                DashIconify(
                                                                    icon="mdi:delete",
                                                                    width=16,
                                                                ),
                                                                id={
                                                                    "type": "delete-playbook-button",
                                                                    "index": playbook_config.yaml_file,
                                                                },
                                                                color="light",
                                                                size="sm",
                                                                title="Delete",
                                                                className="px-2",
                                                            ),
                                                        ],
                                                        size="sm",
                                                        className="w-100",
                                                    ),
                                                ],
                                                className="mx-3 d-flex flex-column halberd-text",
                                                # Add zindex to prevent click propagation on buttons
                                                style={"zIndex": "1"},
                                            ),
                                        ],
                                        width=3,
                                        className="d-flex align-items-center",
                                    ),
                                ],
                                className="g-0",
                            ),
                        ],
                        className="p-3",
                    ),
                ],
                className="mb-3 halberd-depth-card",
                style={
                    "backgroundColor": "#2d2d2d",
                },
            ),
        ],
        # Click handler div
        id={"type": "playbook-card-click", "index": playbook_config.yaml_file},
        className="cursor-pointer hover-highlight",
        # CSS to handle hover and click states
        style={
            "position": "relative",
            "cursor": "pointer",
        },
    )


# Static div for export playbook workflow
export_pb_div = html.Div(
    [
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        # Mask Config Values Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Mask Playbook Config Values",
                                            className="mb-2",
                                        ),
                                        html.Div(
                                            [
                                                daq.BooleanSwitch(
                                                    id="export-playbook-mask-param-boolean",
                                                    on=True,
                                                    color="var(--brand-red)",  # Halberd red color
                                                    className="me-2",
                                                ),
                                                html.Span(
                                                    "Hide sensitive configuration values",
                                                    className="ms-2 align-middle",
                                                ),
                                            ],
                                            className="d-flex align-items-center",
                                        ),
                                    ],
                                    width=12,
                                    className="mb-3",
                                ),
                            ]
                        ),
                        # Export Filename Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Export File Name (Optional)",
                                            className="mb-2",
                                        ),
                                        dbc.Input(
                                            id="export-playbook-filename-text-input",
                                            placeholder="my_playbook_007.yaml",
                                            className="bg-halberd-dark halberd-input halberd-text",
                                        ),
                                        html.Small(
                                            "File will be exported as YAML",
                                            className="text-muted mt-2 d-block",
                                        ),
                                    ],
                                    width=12,
                                    className="mb-4",
                                ),
                            ]
                        ),
                        # Export Button Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            [
                                                DashIconify(
                                                    icon="material-symbols:download-rounded",
                                                    width=24,
                                                    height=24,
                                                    className="me-2",
                                                ),
                                                "Export Playbook",
                                            ],
                                            id="export-playbook-button",
                                            className="float-end halberd-button",
                                            n_clicks=0,
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                    ]
                )
            ],
            className="bg-halberd-dark border-secondary",
        ),
    ],
    className="p-3",
)

# Static div for playbook schedule workflow
schedule_pb_div = html.Div(
    [
        # Card container
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        # Execution Time Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Execution Time *",
                                            html_for="set-time-input",
                                            className="mb-2",
                                        ),
                                        dbc.Input(
                                            id="set-time-input",
                                            type="time",
                                            required=True,
                                            className="bg-halberd-dark halberd-input halberd-text",
                                        ),
                                    ],
                                    width=12,
                                    className="mb-4",
                                ),
                            ]
                        ),
                        # Date Range Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Date Range *",
                                            html_for="automator-date-range-picker",
                                            className="me-2 mb-2",
                                        ),
                                        dcc.DatePickerRange(
                                            id="automator-date-range-picker",
                                            min_date_allowed=date.today(),
                                            max_date_allowed=date(9999, 12, 31),
                                            initial_visible_month=date.today(),
                                            className="bg-halberd-dark halberd-text",
                                        ),
                                    ],
                                    width=12,
                                    className="mb-4",
                                ),
                            ]
                        ),
                        # Repeat Switch Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Repeat Execution", className="mb-2"),
                                        html.Div(
                                            [
                                                daq.BooleanSwitch(
                                                    id="schedule-repeat-boolean",
                                                    on=False,
                                                    color="var(--brand-red)",
                                                    className="me-2",
                                                ),
                                                html.Span(
                                                    "Enable repeat execution",
                                                    className="ms-2 align-middle",
                                                ),
                                            ],
                                            className="d-flex align-items-center",
                                        ),
                                    ],
                                    width=12,
                                    className="mb-4",
                                ),
                            ]
                        ),
                        # Repeat Frequency Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Repeat Frequency",
                                            html_for="repeat-options-dropdown",
                                            className="mb-2",
                                        ),
                                        dcc.Dropdown(
                                            id="repeat-options-dropdown",
                                            options=[
                                                {"label": "Daily", "value": "Daily"},
                                                {"label": "Weekly", "value": "Weekly"},
                                                {
                                                    "label": "Monthly",
                                                    "value": "Monthly",
                                                },
                                            ],
                                            className="bg-halberd-dark halberd-dropdown halberd-text",
                                        ),
                                    ],
                                    width=12,
                                    className="mb-4",
                                ),
                            ]
                        ),
                        # Schedule Name Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Schedule Name (Optional)",
                                            html_for="schedule-name-input",
                                            className="mb-2",
                                        ),
                                        dbc.Input(
                                            id="schedule-name-input",
                                            placeholder="my_schedule",
                                            className="bg-halberd-dark halberd-input halberd-text",
                                        ),
                                    ],
                                    width=12,
                                    className="mb-4",
                                ),
                            ]
                        ),
                        # Schedule Button Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            [
                                                DashIconify(
                                                    icon="material-symbols:schedule-outline",  # Clock icon for schedule
                                                    width=24,
                                                    height=24,
                                                    className="me-2",
                                                ),
                                                "Schedule Playbook",
                                            ],
                                            id="schedule-playbook-button",
                                            n_clicks=0,
                                            className="float-end halberd-button",
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                    ]
                )
            ],
            className="bg-halberd-dark halberd-depth-card halberd-text",
        ),
    ],
    className="p-3",
)


def generate_playbook_creator_offcanvas():
    """Generate off-canvas component for creating new playbooks"""
    return [
        # Playbook metadata form
        dbc.Form(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Playbook Name *",
                                    html_for="pb-name-input-offcanvas",
                                ),
                                dbc.Input(
                                    type="text",
                                    id="pb-name-input-offcanvas",
                                    placeholder="Enter playbook name",
                                    className="bg-halberd-dark halberd-input halberd-text",
                                ),
                            ]
                        )
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Description *", html_for="pb-desc-input-offcanvas"
                                ),
                                dbc.Textarea(
                                    id="pb-desc-input-offcanvas",
                                    placeholder="Enter playbook description",
                                    className="bg-halberd-dark halberd-input halberd-text",
                                ),
                            ]
                        )
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Author *", html_for="pb-author-input-offcanvas"
                                ),
                                dbc.Input(
                                    type="text",
                                    id="pb-author-input-offcanvas",
                                    placeholder="Enter author name",
                                    className="bg-halberd-dark halberd-input halberd-text",
                                ),
                            ]
                        )
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "References", html_for="pb-refs-input-offcanvas"
                                ),
                                dbc.Input(
                                    type="text",
                                    id="pb-refs-input-offcanvas",
                                    placeholder="Enter references (optional)",
                                    className="bg-halberd-dark halberd-input halberd-text",
                                ),
                            ]
                        )
                    ],
                    className="mb-4",
                ),
                # Steps section
                html.H4("Playbook Steps", className="mb-3 halberd-brand-heading"),
                html.Div(
                    id="playbook-steps-container",
                    children=[
                        # Initial step
                        generate_step_form(1)
                    ],
                ),
                # Add step button
                dbc.Button(
                    [html.I(className="bi bi-plus-lg me-2"), "Add Step"],
                    id="add-playbook-step-button",
                    className="mt-3 mb-4 halberd-button-secondary",
                ),
                # Create playbook button
                dbc.Button(
                    [html.I(className="bi bi-save me-2"), "Create Playbook"],
                    id="create-playbook-offcanvas-button",
                    className="w-100 halberd-button",
                ),
            ]
        )
    ]


def generate_step_form(step_number):
    """Generate form elements for a single playbook step"""
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [html.H5(f"Step {step_number}", className="mb-3")],
                                width=10,
                            ),
                            dbc.Col(
                                [
                                    html.Button(
                                        html.I(className="bi bi-trash"),
                                        id={
                                            "type": "remove-step-button",
                                            "index": step_number,
                                        },
                                        className="btn btn-link text-danger",
                                        style={"float": "right"},
                                    )
                                    if step_number > 1
                                    else None
                                ],
                                width=2,
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Module *"),
                                    dcc.Dropdown(
                                        id={
                                            "type": "step-module-dropdown",
                                            "index": step_number,
                                        },
                                        options=[
                                            {"label": technique().name, "value": tid}
                                            for tid, technique in TechniqueRegistry.list_techniques().items()
                                        ],
                                        placeholder="Select module",
                                        className="bg-halberd-dark halberd-text halberd-dropdown",
                                    ),
                                ]
                            )
                        ],
                        className="mb-3",
                    ),
                    # Dynamic parameters section
                    html.Div(
                        id={"type": "step-params-container", "index": step_number}
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Wait (seconds)"),
                                    dbc.Input(
                                        type="number",
                                        id={
                                            "type": "step-wait-input",
                                            "index": step_number,
                                        },
                                        placeholder="0",
                                        min=0,
                                        className="bg-halberd-dark halberd-input halberd-text",
                                    ),
                                ]
                            )
                        ],
                        className="mb-3",
                    ),
                ]
            )
        ],
        className="mb-3 halberd-depth-card",
    )


def generate_playbook_editor_offcanvas():
    return dbc.Offcanvas(
        [
            # Playbook metadata form
            dbc.Form(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Playbook Name *",
                                        html_for="pb-name-input-editor",
                                    ),
                                    dbc.Input(
                                        type="text",
                                        id="pb-name-input-editor",
                                        placeholder="Enter playbook name",
                                        className="bg-halberd-dark halberd-input halberd-text",
                                    ),
                                ]
                            )
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Description *", html_for="pb-desc-input-editor"
                                    ),
                                    dbc.Textarea(
                                        id="pb-desc-input-editor",
                                        placeholder="Enter playbook description",
                                        className="bg-halberd-dark halberd-input halberd-text",
                                    ),
                                ]
                            )
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Author *", html_for="pb-author-input-editor"
                                    ),
                                    dbc.Input(
                                        type="text",
                                        id="pb-author-input-editor",
                                        placeholder="Enter author name",
                                        className="bg-halberd-dark halberd-input halberd-text",
                                    ),
                                ]
                            )
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "References", html_for="pb-refs-input-editor"
                                    ),
                                    dbc.Input(
                                        type="text",
                                        id="pb-refs-input-editor",
                                        placeholder="Enter references (optional)",
                                        className="bg-halberd-dark halberd-input halberd-text",
                                    ),
                                ]
                            )
                        ],
                        className="mb-4",
                    ),
                    # Steps section
                    html.Div(
                        [
                            html.H4(
                                "Playbook Steps", className="mb-3 halberd-brand-heading"
                            ),
                            html.Div(id="playbook-steps-editor-container"),
                            # Add step button
                            dbc.Button(
                                [html.I(className="bi bi-plus-lg me-2"), "Add Step"],
                                id="add-playbook-step-editor-button",
                                color="secondary",
                                className="mt-3 mb-4",
                            ),
                        ]
                    ),
                    # Update playbook button
                    dbc.Button(
                        [html.I(className="bi bi-save me-2"), "Update Playbook"],
                        id="update-playbook-editor-button",
                        className="w-100 halberd-button",
                    ),
                ]
            )
        ],
        id="playbook-editor-offcanvas",
        title=html.H3("Playbook Editor"),
        is_open=False,
        placement="end",
        style={"width": "50%", "max-width": "none"},
        className="bg-halberd-dark halberd-text halberd-offcanvas",
        backdropClassName="halberd-offcanvas-backdrop",
    )


def playbook_editor_create_parameter_inputs(module_id, existing_params=None):
    """Helper function to create parameter input elements"""
    if not module_id:
        return []

    # Initialize existing_params to empty dict if None
    existing_params = existing_params or {}

    technique = TechniqueRegistry.get_technique(module_id)()
    params = technique.get_parameters()

    if not params:
        return html.P("No config required", className="halberd-text")

    param_inputs = []
    for param_name, param_config in params.items():
        required = param_config.get("required", False)
        label_text = f"{param_config['name']} {'*' if required else ''}"

        input_type = param_config.get("input_field_type", "text")

        # Create input with existing value if available
        if input_type == "bool":
            input_elem = daq.BooleanSwitch(
                id={"type": "param-input-editor", "param": param_name},
                on=existing_params.get(param_name, param_config.get("default", False)),
            )
        else:
            input_elem = dbc.Input(
                type=input_type,
                id={"type": "param-input-editor", "param": param_name},
                value=existing_params.get(param_name, param_config.get("default", "")),
                placeholder=param_config.get("default", ""),
                className="bg-halberd-dark halberd-text halberd-input",
            )

        param_inputs.append(
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Label(label_text, className="halberd-text"), input_elem]
                    )
                ],
                className="mb-3",
            )
        )

    return param_inputs


def create_execution_progress_offcanvas():
    """Creates the execution progress off-canvas"""
    return dbc.Offcanvas(
        [
            # Info message
            dbc.Alert(
                [
                    DashIconify(icon="mdi:information", className="me-2"),
                    "You can close this window and return anytime to check progress. ",
                    "Click the 'View Progress' button to reopen.",
                ],
                color="primary",
                className="mb-4",
            ),
            # Progress content
            html.Div(id="playbook-execution-progress", className="mb-4"),
            # Interval for updates
            dcc.Interval(id="execution-interval", interval=1000, disabled=True),
        ],
        id="execution-progress-offcanvas",
        title=html.H3("Execution Progress"),
        placement="end",
        is_open=False,
        style={"width": "50%"},
        className="bg-halberd-dark halberd-offcanvas",
        backdropClassName="halberd-offcanvas-backdrop",
        scrollable=True,
    )


def create_step_progress_card(
    step_number, module_name, status=None, is_active=False, message=None
):
    """Creates a card showing execution status for a single playbook step"""
    # Define status icon and color
    if is_active:
        icon = DashIconify(
            icon="mdi:progress-clock", width=24, className="text-primary animate-spin"
        )
        status_color = "text-light"
    elif status == "success":
        icon = DashIconify(icon="mdi:check-circle", width=24, className="text-success")
        status_color = "text-success"
    elif status == "failed":
        icon = DashIconify(icon="mdi:alert-circle", width=24, className="text-danger")
        status_color = "text-danger"
    else:
        icon = DashIconify(
            icon="mdi:circle-outline", width=24, className="text-gray-400"
        )
        status_color = "text-muted"

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(icon, width=1),
                            dbc.Col(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.H6(
                                                    f"Step {step_number}: {module_name}",
                                                    className="mb-0 halberd-text",
                                                ),
                                                width=9,
                                            ),
                                            dbc.Col(
                                                html.Span(
                                                    status.title()
                                                    if status
                                                    else "Pending",
                                                    className=status_color,
                                                ),
                                                width=3,
                                                className="text-end",
                                            ),
                                        ]
                                    ),
                                    html.Small(message, className="text-danger")
                                    if message
                                    else None,
                                ],
                                width=11,
                            ),
                        ],
                        className="align-items-center halberd-text",
                    )
                ]
            )
        ],
        className=f"mb-2 {'border-primary' if is_active else ''} bg-halberd-dark",
    )


# Create Automator layout
layout = create_playbook_manager_layout

# Callbacks
"""Callback to generate attack sequence visualization in Automator"""


@callback(
    Output("playbook-visualization-container", "children"),
    [Input({"type": "playbook-card-click", "index": ALL}, "n_clicks")],
    prevent_initial_call=True,
)
def update_visualization(n_clicks):
    """Update the visualization when a playbook is selected"""
    if not callback_context.triggered:
        raise PreventUpdate

    # Get the triggered component's ID
    triggered = callback_context.triggered[0]
    prop_id = json.loads(triggered["prop_id"].rsplit(".", 1)[0])

    if triggered["value"] is None:  # No clicks yet
        raise PreventUpdate

    playbook_id = prop_id["index"]

    try:
        pb_config = Playbook(playbook_id)
        # Return both the visualization and some playbook info
        return html.Div(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.Div(
                                f"Playbook : {pb_config.name}",
                                className="mb-0 halberd-brand text-2xl",
                            )
                        ),
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Description:", className="mb-2 halberd-typography"
                                ),
                                html.P(
                                    pb_config.description, className="mb-3 halberd-text"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.P(
                                                f"Total Steps: {pb_config.steps}",
                                                className="mb-1 halberd-depth-card",
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            html.P(
                                                f"Author: {pb_config.author}",
                                                className="mb-1 halberd-depth-card",
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            html.P(
                                                f"Created: {pb_config.creation_date}",
                                                className="mb-1 halberd-depth-card",
                                            ),
                                            md=4,
                                        ),
                                    ],
                                    style={"textAlign": "center"},
                                ),
                            ]
                        ),
                    ],
                    className="bg-halberd-dark halberd-depth-card",
                ),
                html.Div(playbook_viz_generator(pb_config.name), className="mb-3"),
            ]
        )
    except Exception as e:
        return html.Div(
            [
                html.H4("Error Loading Visualization", className="text-danger"),
                html.P(str(e), className="text-muted"),
            ],
            className="p-3",
        )


"""Callback to execute attack sequence in automator view"""


@callback(
    Output("execution-progress-offcanvas", "is_open", allow_duplicate=True),
    Output("app-notification", "is_open", allow_duplicate=True),
    Output("app-notification", "children", allow_duplicate=True),
    Output("app-error-display-modal", "is_open", allow_duplicate=True),
    Output("app-error-display-modal-body", "children", allow_duplicate=True),
    Output("selected-playbook-data", "data", allow_duplicate=True),
    Output("execution-interval", "disabled", allow_duplicate=True),
    Input({"type": "execute-playbook-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def execute_playbook_callback(n_clicks):
    """Execute playbook and initialize progress tracking"""
    if not any(n_clicks):
        raise PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Get clicked playbook
    button_id = ctx.triggered[0]["prop_id"].rsplit(".", 1)[0]
    playbook_file = eval(button_id)["index"]

    try:
        # Execute playbook in background thread
        def execute_playbook():
            playbook = Playbook(playbook_file)
            playbook.execute()

        execution_thread = threading.Thread(target=execute_playbook)
        execution_thread.daemon = True
        execution_thread.start()

        return True, True, "Playbook Execution Started", False, "", playbook_file, False

    except PlaybookError as e:
        error_msg = f"Playbook Execution Failed: {str(e.message)}"
        return False, False, "", True, error_msg, None, True
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        return False, False, "", True, error_msg, None, True


"""Callback to open attack scheduler off canvas"""


@callback(
    Output(
        component_id="automator-offcanvas",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="automator-offcanvas",
        component_property="title",
        allow_duplicate=True,
    ),
    Output(
        component_id="automator-offcanvas",
        component_property="children",
        allow_duplicate=True,
    ),
    Output(
        component_id="selected-playbook-data",
        component_property="data",
        allow_duplicate=True,
    ),
    Input({"type": "open-schedule-win-playbook-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def toggle_pb_schedule_canvas_callback(n_clicks):
    if not any(n_clicks):
        raise PreventUpdate

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Extract playbook name from context
    button_id = ctx.triggered[0]["prop_id"].rsplit(".", 1)[0]
    selected_pb_name = eval(button_id)["index"]

    return True, html.H3(["Schedule Playbook"]), schedule_pb_div, selected_pb_name


"""Callback to create new automator schedule"""


@callback(
    Output(
        component_id="app-notification",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-notification",
        component_property="children",
        allow_duplicate=True,
    ),
    Output(
        component_id="automator-offcanvas",
        component_property="is_open",
        allow_duplicate=True,
    ),
    State(component_id="selected-playbook-data", component_property="data"),
    State(component_id="set-time-input", component_property="value"),
    State(component_id="automator-date-range-picker", component_property="start_date"),
    State(component_id="automator-date-range-picker", component_property="end_date"),
    State(component_id="schedule-repeat-boolean", component_property="on"),
    State(component_id="repeat-options-dropdown", component_property="value"),
    State(component_id="schedule-name-input", component_property="value"),
    Input(component_id="schedule-playbook-button", component_property="n_clicks"),
    prevent_initial_call=True,
)
def create_new_schedule_callback(
    selected_pb_data,
    execution_time,
    start_date,
    end_date,
    repeat_flag,
    repeat_frequency,
    schedule_name,
    n_clicks,
):
    if n_clicks == 0:
        raise PreventUpdate

    if selected_pb_data is None:
        raise PreventUpdate

    playbook_id = selected_pb_data
    # Create new schedule
    AddNewSchedule(
        schedule_name,
        playbook_id,
        start_date,
        end_date,
        execution_time,
        repeat_flag,
        repeat_frequency,
    )

    # Send notification after new schedule is created and close scheduler off canvas
    return True, "Playbook Scheduled", False


"""Callback to export playbook"""


@callback(
    Output(
        component_id="app-download-sink",
        component_property="data",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-notification",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-notification",
        component_property="children",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-error-display-modal",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-error-display-modal-body",
        component_property="children",
        allow_duplicate=True,
    ),
    State(component_id="selected-playbook-data", component_property="data"),
    State(component_id="export-playbook-mask-param-boolean", component_property="on"),
    State(
        component_id="export-playbook-filename-text-input", component_property="value"
    ),
    Input(component_id="export-playbook-button", component_property="n_clicks"),
    prevent_initial_call=True,
)
def export_playbook_callback(selected_pb_data, mask_param, export_file_name, n_clicks):
    if n_clicks == 0:
        raise PreventUpdate

    playbook_file = selected_pb_data
    playbook = Playbook(playbook_file)

    if not export_file_name:
        export_file_base_name = "Halberd_Playbook"  # Set default file name
        export_file_name = (
            export_file_base_name + "-" + (playbook.name).replace(" ", "_") + ".yml"
        )

    # Export playbook
    playbook_export_file_path = playbook.export(
        export_file=export_file_name, include_params=not (mask_param)
    )

    # Download playbook and send app notification
    return (
        dcc.send_file(playbook_export_file_path),
        True,
        "Playbook Exported",
        False,
        "",
    )


"""Callback to import playbook"""


@callback(
    Output(
        component_id="app-notification",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-notification",
        component_property="children",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-error-display-modal",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-error-display-modal-body",
        component_property="children",
        allow_duplicate=True,
    ),
    Output("playbook-list-container", "children", allow_duplicate=True),
    Output("playbook-stats", "children", allow_duplicate=True),
    Input(component_id="import-pb-button", component_property="n_clicks"),
    Input(component_id="upload-playbook", component_property="contents"),
    prevent_initial_call=True,
)
def import_playbook_callback(n_clicks, file_contents):
    if n_clicks == 0:
        raise PreventUpdate

    if file_contents:
        try:
            # Import playbook
            Playbook.import_playbook(file_contents)

            # Refresh the playbook list
            playbooks = GetAllPlaybooks()
            playbook_items = []

            for pb_file in playbooks:
                try:
                    pb_config = Playbook(pb_file)
                    # Apply search filter if query exists
                    playbook_items.append(create_playbook_item(pb_config))
                except Exception as e:
                    print(f"Error loading playbook {pb_file}: {str(e)}")

            # Generate stats
            stats = get_playbook_stats()
            stats_text = (
                f"{stats['total_playbooks']} playbooks loaded • "
                f"Last sync: {stats['last_sync'].strftime('%I:%M %p') if stats['last_sync'] else 'never'}"
            )

            # Import success - display notification and update playbook list
            return True, "Playbook Imported", False, "", playbook_items, stats_text
        except Exception as e:
            # Display error in modal pop up
            return False, "", True, str(e), no_update, no_update
    else:
        raise PreventUpdate


"""Callback to open playbook creator off canvas"""


@callback(
    Output(
        component_id="automator-offcanvas",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="automator-offcanvas",
        component_property="title",
        allow_duplicate=True,
    ),
    Output(
        component_id="automator-offcanvas",
        component_property="children",
        allow_duplicate=True,
    ),
    Input(
        component_id="open-creator-win-playbook-button", component_property="n_clicks"
    ),
    prevent_initial_call=True,
)
def toggle_pb_creator_canvas_callback(n_clicks):
    if n_clicks:
        return (
            True,
            [html.H3("Create New Playbook")],
            generate_playbook_creator_offcanvas(),
        )

    raise PreventUpdate


"""Callback to create new playbook"""


@callback(
    Output(
        component_id="playbook-creator-modal",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-notification",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-notification",
        component_property="children",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-error-display-modal",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="app-error-display-modal-body",
        component_property="children",
        allow_duplicate=True,
    ),
    State(component_id="pb-name-input", component_property="value"),
    State(component_id="pb-desc-input", component_property="value"),
    State(component_id="pb-author-input", component_property="value"),
    State(component_id="pb-refs-input", component_property="value"),
    Input(component_id="create-playbook-button", component_property="n_clicks"),
    prevent_initial_call=True,
)
def create_new_pb_callback(pb_name, pb_desc, pb_author, pb_references, n_clicks):
    if n_clicks == 0:
        raise PreventUpdate

    try:
        new_playbook = Playbook.create_new(
            name=pb_name,
            author=pb_author,
            description=pb_desc,
            references=[pb_references],
        )
        return False, True, f"New Playbook Created : {new_playbook.name}", False, ""
    except Exception as e:
        return True, False, "", True, str(e)


"""Callback to display technique info from playbook node in modal"""


@callback(
    Output(
        component_id="app-technique-info-display-modal-body",
        component_property="children",
    ),
    Output(
        component_id="app-technique-info-display-modal", component_property="is_open"
    ),
    Input(
        component_id="auto-attack-sequence-cytoscape-nodes",
        component_property="tapNodeData",
    ),
    [
        State(
            component_id="app-technique-info-display-modal",
            component_property="is_open",
        )
    ],
    prevent_initial_call=True,
)
def toggle_t_info_modal_callback(data, is_open):
    if data:
        # Extract module_id from node label
        if data["label"] != "None":
            info = data["info"]
        else:
            raise PreventUpdate

        if info == "time":
            # Display time gap
            wait_time = data["label"]
            return [html.B(f"Time Gap : {wait_time} seconds")], True
        else:
            # Display module info
            pb_step_info = data["info"]
            step_data = next(iter(pb_step_info.items()))
            module_id = step_data[1]["Module"]
            return generate_technique_info(module_id), not is_open
    else:
        raise PreventUpdate


"""Callback to open/close add to playbook modal on Attack page"""


@callback(
    Output(component_id="add-to-playbook-modal", component_property="is_open"),
    [
        Input(
            component_id="open-add-to-playbook-modal-button",
            component_property="n_clicks",
        ),
        Input(
            component_id="close-add-to-playbook-modal-button",
            component_property="n_clicks",
        ),
        Input(
            component_id="confirm-add-to-playbook-modal-button",
            component_property="n_clicks",
        ),
    ],
    [State(component_id="add-to-playbook-modal", component_property="is_open")],
    prevent_initial_call=True,
)
def toggle_add_to_pb_modal_callback(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    return is_open


"""[Automator] Callback to generate/update playbook list in automator"""


@callback(
    Output("playbook-list-container", "children"),
    Output("playbook-stats", "children"),
    Input("playbook-search", "value"),
)
def update_playbook_list_callback(search_query):
    """Update the playbook list and stats based on search query"""
    # Get all available playbooks on system
    playbooks = GetAllPlaybooks()

    # Generate stats
    stats = get_playbook_stats()
    stats_text = (
        f"{stats['total_playbooks']} playbooks loaded • "
        f"Last sync: {stats['last_sync'].strftime('%I:%M %p') if stats['last_sync'] else 'never'}"
    )

    # If no playbooks found on system
    if not playbooks:
        empty_playbook_list_div = html.Div(
            children=[
                html.Div(
                    [
                        DashIconify(
                            icon="mdi:information-outline",  # Information icon
                            width=48,
                            height=48,
                            className="text-muted mb-3",
                        ),
                        html.P(
                            "Create or Import a playbook",  # Default message when no playbook is selected
                            className="halberd-text text-muted",
                        ),
                    ],
                    className="text-center",
                )
            ],
            className="d-flex justify-content-center align-items-center",
            style={"padding": "20px"},
        )
        return empty_playbook_list_div, stats_text

    # Initialize list to store playbook items
    playbook_items = []

    for pb_file in playbooks:
        try:
            pb_config = Playbook(pb_file)
            # Apply search filter if query exists
            if search_query and search_query.lower() not in pb_config.name.lower():
                continue
            playbook_items.append(create_playbook_item(pb_config))
        except Exception as e:
            print(f"Error loading playbook {pb_file}: {str(e)}")

    return playbook_items, stats_text


"""Callback to delete playbook from automator"""


@callback(
    Output("playbook-list-container", "children", allow_duplicate=True),
    Output("playbook-stats", "children", allow_duplicate=True),
    Input({"type": "delete-playbook-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def delete_playbook(n_clicks):
    """Handles playbook deletion"""
    if not any(n_clicks):
        return no_update

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    button_id = ctx.triggered[0]["prop_id"].rsplit(".", 1)[0]
    playbook_file = eval(button_id)["index"]

    try:
        # Delete the playbook file
        os.remove(os.path.join(AUTOMATOR_PLAYBOOKS_DIR, playbook_file))

        # Refresh the playbook list
        playbooks = GetAllPlaybooks()

        # Generate stats
        stats = get_playbook_stats()
        stats_text = (
            f"{stats['total_playbooks']} playbooks loaded • "
            f"Last sync: {stats['last_sync'].strftime('%I:%M %p') if stats['last_sync'] else 'never'}"
        )

        if not playbooks:
            empty_playbook_list_div = html.Div(
                children=[
                    html.Div(
                        [
                            DashIconify(
                                icon="mdi:information-outline",  # Information icon
                                width=48,
                                height=48,
                                className="text-muted mb-3",
                            ),
                            html.P(
                                "Create or Import a playbook",  # Default message when no playbook is selected
                                className="text-muted",
                            ),
                        ],
                        className="text-center",
                    )
                ],
                className="d-flex justify-content-center align-items-center",
                style={"padding": "20px"},
            )
            return empty_playbook_list_div, stats_text

        # Initialize list to store playbook items
        playbook_items = []

        for pb_file in playbooks:
            try:
                pb_config = Playbook(pb_file)
                # Apply search filter if query exists
                playbook_items.append(create_playbook_item(pb_config))
            except Exception as e:
                print(f"Error loading playbook {pb_file}: {str(e)}")

        return playbook_items, stats_text
    except Exception as e:
        print(f"Error deleting playbook {playbook_file}: {str(e)}")
        return no_update


"""Callback to close the playbook information modal"""


@callback(
    Output("automator-playbook-info-display-modal", "is_open", allow_duplicate=True),
    Input("close-automator-playbook-info-display-modal", "n_clicks"),
    State("automator-playbook-info-display-modal", "is_open"),
    prevent_initial_call=True,
)
def close_pb_info_modal_callback(n_clicks, is_open):
    if n_clicks:
        return False
    return is_open


"""Callback to open playbook export modal"""


@callback(
    Output(
        component_id="automator-offcanvas",
        component_property="is_open",
        allow_duplicate=True,
    ),
    Output(
        component_id="automator-offcanvas",
        component_property="title",
        allow_duplicate=True,
    ),
    Output(
        component_id="automator-offcanvas",
        component_property="children",
        allow_duplicate=True,
    ),
    Output(
        component_id="selected-playbook-data",
        component_property="data",
        allow_duplicate=True,
    ),
    Input({"type": "open-export-win-playbook-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def toggle_pb_export_canvas_callback(n_clicks):
    if not any(n_clicks):
        raise PreventUpdate

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Extract playbook name from context
    button_id = ctx.triggered[0]["prop_id"].rsplit(".", 1)[0]
    selected_pb_name = eval(button_id)["index"]

    return True, [html.H3("Export Playbook")], export_pb_div, selected_pb_name


"""Create new playbook functionality callbacks"""
"""[Playbook Creator] Callback to generate/update parameter fields from selected technique"""


@callback(
    Output({"type": "step-params-container", "index": MATCH}, "children"),
    Input({"type": "step-module-dropdown", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def update_step_parameters(module_id):
    """Update parameter fields based on selected module"""
    if not module_id:
        return []

    technique = TechniqueRegistry.get_technique(module_id)()
    params = technique.get_parameters()

    if not params:
        return html.P("No parameters required", className="text-muted")

    param_inputs = []
    for param_name, param_config in params.items():
        required = param_config.get("required", False)
        label_text = f"{param_config['name']} {'*' if required else ''}"

        input_type = param_config.get("input_field_type", "text")

        # Create the appropriate input element
        if input_type == "bool":
            input_elem = daq.BooleanSwitch(
                id={"type": "param-input", "param": param_name},
                on=param_config.get("default", False),
            )
        else:
            # Add any input validation based on technique requirements
            input_props = {
                "type": input_type,
                "id": {"type": "param-input", "param": param_name},
                "placeholder": param_config.get("default", ""),
                "className": "bg-halberd-dark text-light",
                "required": required,
            }

            # Add any additional validation attributes
            if input_type == "number":
                input_props.update(
                    {
                        "min": param_config.get("min", None),
                        "max": param_config.get("max", None),
                        "step": param_config.get("step", None),
                    }
                )

            input_elem = dbc.Input(**input_props)

        # Add description or help text if available
        help_text = None
        if param_config.get("description"):
            help_text = html.Small(
                param_config["description"], className="text-muted d-block mt-1"
            )

        param_inputs.append(
            dbc.Row(
                [dbc.Col([dbc.Label(label_text), input_elem, help_text])],
                className="mb-3",
            )
        )

    return param_inputs


"""[Playbook Creator] Callback to add a new step in playbook"""


@callback(
    Output("playbook-steps-container", "children"),
    Input("add-playbook-step-button", "n_clicks"),
    State("playbook-steps-container", "children"),
    prevent_initial_call=True,
)
def add_playbook_step(n_clicks, current_steps):
    """Add a new step form to the playbook creator"""
    if n_clicks:
        new_step_number = len(current_steps) + 1
        return current_steps + [generate_step_form(new_step_number)]
    return current_steps


"""[Playbook Creator] Callback to remove a step from playbook"""


@callback(
    Output("playbook-steps-container", "children", allow_duplicate=True),
    Input({"type": "remove-step-button", "index": ALL}, "n_clicks"),
    State("playbook-steps-container", "children"),
    prevent_initial_call=True,
)
def remove_playbook_step(n_clicks, current_steps):
    """Remove a step from the playbook creator"""
    if not any(n_clicks):
        raise PreventUpdate

    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = json.loads(ctx.triggered[0]["prop_id"].rsplit(".")[0])
    step_to_remove = button_id["index"]

    # Remove the step and renumber remaining steps
    remaining_steps = [
        step
        for step in current_steps
        if int(
            step["props"]["children"][0]["props"]["children"][0]["props"]["children"][
                0
            ]["props"]["children"][0]["props"]["children"].split()[-1]
        )
        != step_to_remove
    ]
    renumbered_steps = [generate_step_form(i + 1) for i in range(len(remaining_steps))]

    return renumbered_steps


"""[Playbook Creator] Callback to create a new playbook from offcanvas configuration"""


@callback(
    Output("app-notification", "is_open", allow_duplicate=True),
    Output("app-notification", "children", allow_duplicate=True),
    Output("app-error-display-modal", "is_open", allow_duplicate=True),
    Output("app-error-display-modal-body", "children", allow_duplicate=True),
    Output("automator-offcanvas", "is_open", allow_duplicate=True),
    Output("playbook-list-container", "children", allow_duplicate=True),
    Output("playbook-stats", "children", allow_duplicate=True),
    Input("create-playbook-offcanvas-button", "n_clicks"),
    [
        State("pb-name-input-offcanvas", "value"),
        State("pb-desc-input-offcanvas", "value"),
        State("pb-author-input-offcanvas", "value"),
        State("pb-refs-input-offcanvas", "value"),
        State({"type": "step-module-dropdown", "index": ALL}, "value"),
        State({"type": "step-wait-input", "index": ALL}, "value"),
        State({"type": "param-input", "param": ALL}, "value"),
        State({"type": "param-input", "param": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def create_playbook_from_offcanvas(
    n_clicks, name, desc, author, refs, modules, waits, param_values, param_ids
):
    """Create a new playbook from the off-canvas form data"""
    if not n_clicks:
        raise PreventUpdate

    try:
        # Validate required fields
        if not all([name, desc, author]):
            raise ValueError("Please fill in all required fields")

        if not any(modules):
            raise ValueError("At least one step is required")

        # Create new playbook
        new_playbook = Playbook.create_new(
            name=name,
            author=author,
            description=desc,
            references=[refs] if refs else None,
        )

        # Group parameters by step
        step_params = {}
        for i, module in enumerate(modules):
            if module:  # If module is selected
                # Get technique parameters configuration
                technique = TechniqueRegistry.get_technique(module)()
                technique_params = technique.get_parameters()

                # Initialize params dict for this step
                step_params[i] = {}

                # Match parameters with their values for this step's technique
                for param_id, param_value in zip(param_ids, param_values):
                    param_name = param_id["param"]
                    if param_name in technique_params:
                        # Convert empty strings to None for optional parameters
                        if param_value == "" and not technique_params[param_name].get(
                            "required", False
                        ):
                            param_value = None
                        step_params[i][param_name] = param_value

        # Add steps with their parameters
        for i, (module, wait) in enumerate(zip(modules, waits)):
            if module:  # Only add steps with selected modules
                new_step = PlaybookStep(
                    module=module,
                    params=step_params.get(i, {}),  # Get parameters for this step
                    wait=int(wait) if wait else 0,
                )
                new_playbook.add_step(new_step, i + 1)

        # get updated list of available playbooks
        playbooks = GetAllPlaybooks()
        playbook_items = []

        for pb_file in playbooks:
            try:
                pb_config = Playbook(pb_file)
                # Apply search filter if query exists
                playbook_items.append(create_playbook_item(pb_config))
            except Exception as e:
                print(f"Error loading playbook {pb_file}: {str(e)}")

        stats = get_playbook_stats()
        stats_text = (
            f"{stats['total_playbooks']} playbooks loaded • "
            f"Last sync: {stats['last_sync'].strftime('%I:%M %p') if stats['last_sync'] else 'never'}"
        )

        return (
            True,
            f"New Playbook Created: {name}",
            False,
            "",
            False,
            playbook_items,
            stats_text,
        )

    except Exception as e:
        return False, "", True, str(e), False, no_update, no_update


"""Playbook editor callbacks"""
"""[Playbook Editor] Callback to open playbook editor off canvas"""


@callback(
    Output("playbook-editor-offcanvas", "is_open", allow_duplicate=True),
    Output(
        component_id="selected-playbook-data-editor-memory-store",
        component_property="data",
        allow_duplicate=True,
    ),
    Input({"type": "edit-playbook-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_editable_playbook_view(n_clicks):
    if not any(n_clicks):
        raise PreventUpdate

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Extract playbook file name from context
    button_id = ctx.triggered[0]["prop_id"].rsplit(".", 1)[0]
    selected_pb = eval(button_id)["index"]

    return True, selected_pb


"""[Playbook Editor] Callback to load & display existing playbook information"""


@callback(
    [
        Output("pb-name-input-editor", "value"),
        Output("pb-desc-input-editor", "value"),
        Output("pb-author-input-editor", "value"),
        Output("pb-refs-input-editor", "value"),
        Output("playbook-steps-editor-container", "children"),
    ],
    Input({"type": "edit-playbook-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def load_playbook_data(n_clicks):
    """Load existing playbook data into editor when opened"""
    if not n_clicks:
        raise PreventUpdate

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Extract playbook file name from context
    button_id = ctx.triggered[0]["prop_id"].rsplit(".", 1)[0]
    selected_pb = eval(button_id)["index"]

    # Find the selected playbook
    try:
        playbook = Playbook(selected_pb)

        # Generate step forms with existing data
        steps = []
        for step_no, step_data in playbook.data["PB_Sequence"].items():
            step_form = dbc.Card(
                [
                    dbc.CardBody(
                        [
                            # Step header
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H5(
                                                f"Step {step_no}",
                                                className="mb-3 text-success",
                                            )
                                        ],
                                        width=10,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Button(
                                                html.I(className="bi bi-trash"),
                                                id={
                                                    "type": "remove-step-editor-button",
                                                    "index": step_no,
                                                },
                                                className="btn btn-link text-danger",
                                                style={"float": "right"},
                                            )
                                            if int(step_no) > 1
                                            else None
                                        ],
                                        width=2,
                                    ),
                                ]
                            ),
                            # Module selector
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Module *"),
                                            dcc.Dropdown(
                                                id={
                                                    "type": "step-module-dropdown-editor",
                                                    "index": step_no,
                                                },
                                                options=[
                                                    {
                                                        "label": technique().name,
                                                        "value": tid,
                                                    }
                                                    for tid, technique in TechniqueRegistry.list_techniques().items()
                                                ],
                                                value=step_data.get("Module"),
                                                placeholder="Select module",
                                                className="bg-halberd-dark halberd-dropdown halberd-text",
                                            ),
                                        ]
                                    )
                                ],
                                className="mb-3",
                            ),
                            # Parameters container
                            html.Div(
                                # Create parameter inputs if module data available
                                playbook_editor_create_parameter_inputs(
                                    step_data.get("Module"), step_data.get("Params", {})
                                )
                                if step_data.get("Module")
                                else [],
                                id={
                                    "type": "step-params-container-editor",
                                    "index": step_no,
                                },
                            ),
                            # Wait time input
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Wait (seconds)"),
                                            dbc.Input(
                                                type="number",
                                                id={
                                                    "type": "step-wait-input-editor",
                                                    "index": step_no,
                                                },
                                                value=step_data.get("Wait", 0),
                                                placeholder="0",
                                                min=0,
                                                className="bg-halberd-dark halberd-text halberd-input",
                                            ),
                                        ]
                                    )
                                ],
                                className="mb-3",
                            ),
                        ]
                    )
                ],
                className="mb-3 halberd-depth-card",
            )
            steps.append(step_form)

        return (
            playbook.name,
            playbook.description,
            playbook.author,
            ", ".join(playbook.references) if playbook.references else "",
            steps,
        )
    except:
        raise PreventUpdate


"""[Playbook Editor] Callback to add a new step in existing playbook"""


@callback(
    Output("playbook-steps-editor-container", "children", allow_duplicate=True),
    Input("add-playbook-step-editor-button", "n_clicks"),
    State("playbook-steps-editor-container", "children"),
    prevent_initial_call=True,
)
def add_playbook_step_editor(n_clicks, current_steps):
    """Add a new step form to the playbook editor"""
    if n_clicks:
        new_step_number = len(current_steps) + 1
        new_step = dbc.Card(
            [
                dbc.CardBody(
                    [
                        # Step header
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H5(
                                            f"Step {new_step_number}",
                                            className="mb-3 text-success",
                                        )
                                    ],
                                    width=10,
                                ),
                                dbc.Col(
                                    [
                                        html.Button(
                                            html.I(className="bi bi-trash"),
                                            id={
                                                "type": "remove-step-editor-button",
                                                "index": new_step_number,
                                            },
                                            className="btn btn-link text-danger",
                                            style={"float": "right"},
                                        )
                                    ],
                                    width=2,
                                ),
                            ]
                        ),
                        # Module selector
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Module *"),
                                        dcc.Dropdown(
                                            id={
                                                "type": "step-module-dropdown-editor",
                                                "index": new_step_number,
                                            },
                                            options=[
                                                {
                                                    "label": technique().name,
                                                    "value": tid,
                                                }
                                                for tid, technique in TechniqueRegistry.list_techniques().items()
                                            ],
                                            placeholder="Select module",
                                            className="bg-halberd-dark halberd-dropdown halberd-text",
                                        ),
                                    ]
                                )
                            ],
                            className="mb-3",
                        ),
                        # Wait time input
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Wait (seconds)"),
                                        dbc.Input(
                                            type="number",
                                            id={
                                                "type": "step-wait-input-editor",
                                                "index": new_step_number,
                                            },
                                            placeholder="0",
                                            min=0,
                                            value=0,
                                            className="bg-halberd-dark halberd-input",
                                        ),
                                    ]
                                )
                            ],
                            className="mb-3",
                        ),
                        # Parameters container (initially empty)
                        html.Div(
                            id={
                                "type": "step-params-container-editor",
                                "index": new_step_number,
                            }
                        ),
                    ]
                )
            ],
            className="mb-3 halberd-depth-card",
        )

        return current_steps + [new_step]
    return current_steps


"""[Playbook Editor] Callback to update parameters on technique change from dropdown"""


@callback(
    Output({"type": "step-params-container-editor", "index": MATCH}, "children"),
    Input({"type": "step-module-dropdown-editor", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def update_step_parameters_editor(module_id):
    """Update parameter fields when module selection changes"""
    if not module_id:
        return []

    return playbook_editor_create_parameter_inputs(module_id)


@callback(
    Output("app-notification", "is_open", allow_duplicate=True),
    Output("app-notification", "children", allow_duplicate=True),
    Output("app-error-display-modal", "is_open", allow_duplicate=True),
    Output("app-error-display-modal-body", "children", allow_duplicate=True),
    Output("playbook-editor-offcanvas", "is_open", allow_duplicate=True),
    Input("update-playbook-editor-button", "n_clicks"),
    [
        State("pb-name-input-editor", "value"),
        State("pb-desc-input-editor", "value"),
        State("pb-author-input-editor", "value"),
        State("pb-refs-input-editor", "value"),
        State({"type": "step-module-dropdown-editor", "index": ALL}, "value"),
        State({"type": "step-wait-input-editor", "index": ALL}, "value"),
        State({"type": "param-input-editor", "param": ALL}, "value"),
        State({"type": "param-input-editor", "param": ALL}, "id"),
        State("selected-playbook-data-editor-memory-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_playbook_from_editor(
    n_clicks,
    name,
    desc,
    author,
    refs,
    modules,
    waits,
    param_values,
    param_ids,
    selected_playbook,
):
    """Update existing playbook from editor data"""
    if not n_clicks:
        raise PreventUpdate

    try:
        # Find the selected playbook
        playbook = Playbook(selected_playbook)
        # Update playbook metadata
        playbook.data["PB_Name"] = name
        playbook.data["PB_Description"] = desc
        playbook.data["PB_Author"] = author
        playbook.data["PB_References"] = (
            [ref.strip() for ref in refs.split(",")] if refs else []
        )

        # Clear existing sequence
        playbook.data["PB_Sequence"] = {}

        # Group parameters by step
        step_params = {}
        for i, module in enumerate(modules):
            if module:
                technique = TechniqueRegistry.get_technique(module)()
                technique_params = technique.get_parameters()
                step_params[i] = {}

                for param_id, param_value in zip(param_ids, param_values):
                    param_name = param_id["param"]
                    if param_name in technique_params:
                        if param_value == "" and not technique_params[param_name].get(
                            "required", False
                        ):
                            param_value = None
                        step_params[i][param_name] = param_value

        # Add updated steps
        for i, (module, wait) in enumerate(zip(modules, waits)):
            if module:
                playbook.data["PB_Sequence"][i + 1] = {
                    "Module": module,
                    "Params": step_params.get(i, {}),
                    "Wait": int(wait) if wait else 0,
                }

        # Save updated playbook
        playbook.save()
        return True, f"Playbook Updated: {name}", False, "", False

    except Exception as e:
        return False, "", True, str(e), False


"""[Playbook Editor] Callback to remove step from playbook and update the playbook steps"""


@callback(
    Output("playbook-steps-editor-container", "children", allow_duplicate=True),
    Input({"type": "remove-step-editor-button", "index": ALL}, "n_clicks"),
    State("playbook-steps-editor-container", "children"),
    prevent_initial_call=True,
)
def remove_playbook_step_editor(n_clicks, current_steps):
    """Remove a step from the playbook editor and renumber remaining steps"""
    if not any(n_clicks) or not current_steps:
        raise PreventUpdate

    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    try:
        button_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
        step_to_remove = button_id["index"]

        # Create new list without the removed step
        remaining_steps = []
        new_step_number = 1

        for step in current_steps:
            # Extract current step number from the card
            current_step_header = step["props"]["children"]["props"]["children"][0][
                "props"
            ]["children"][0]["props"]["children"]["children"]
            current_step_num = int(current_step_header.split()[1])

            if current_step_num != step_to_remove:
                # Update step number in header
                step["props"]["children"]["props"]["children"][0]["props"]["children"][
                    0
                ]["props"]["children"]["children"] = f"Step {new_step_number}"

                # Update all component IDs that contain step number
                for component in [
                    {
                        "type": "remove-step-editor-button",
                        "location": [
                            0,
                            "props",
                            "children",
                            1,
                            "props",
                            "children",
                            "props",
                            "id",
                        ],
                    },
                    {
                        "type": "step-module-dropdown-editor",
                        "location": [
                            1,
                            "props",
                            "children",
                            0,
                            "props",
                            "children",
                            1,
                            "props",
                            "id",
                        ],
                    },
                    {
                        "type": "step-wait-input-editor",
                        "location": [
                            2,
                            "props",
                            "children",
                            0,
                            "props",
                            "children",
                            1,
                            "props",
                            "id",
                        ],
                    },
                    {
                        "type": "step-params-container-editor",
                        "location": [3, "props", "id"],
                    },
                ]:
                    try:
                        # Navigate to the component's location
                        current = step["props"]["children"]["props"]["children"]
                        for loc in component["location"][:-1]:
                            current = current[loc]
                        # Update the ID
                        current[component["location"][-1]]["index"] = new_step_number
                    except (KeyError, IndexError, TypeError):
                        continue

                remaining_steps.append(step)
                new_step_number += 1

        return remaining_steps
    except Exception as e:
        print(f"Error in remove_playbook_step_editor: {str(e)}")
        raise PreventUpdate


"""[Playbook Progress Tracker] Callback to update the execution progress display"""


@callback(
    Output("playbook-execution-progress", "children"),
    Output("execution-interval", "disabled"),
    Input("execution-interval", "n_intervals"),
    State("selected-playbook-data", "data"),
    prevent_initial_call=True,
)
def update_execution_progress(n_intervals, playbook_data):
    """Update the execution progress display"""
    if not playbook_data:
        raise PreventUpdate

    try:
        # Get playbook config
        playbook = Playbook(playbook_data)
        total_steps = len(playbook.data["PB_Sequence"])

        # Get latest execution folder
        execution_folders = [
            d
            for d in os.listdir(AUTOMATOR_OUTPUT_DIR)
            if d.startswith(f"{playbook.name}_")
        ]

        if not execution_folders:
            raise PreventUpdate

        latest_folder = max(execution_folders)
        execution_folder = os.path.join(AUTOMATOR_OUTPUT_DIR, latest_folder)

        # Get execution results
        results = parse_execution_report(execution_folder)
        active_step = len(results)

        # Create status cards for each step
        step_cards = []
        for step_no, step_data in playbook.data["PB_Sequence"].items():
            step_index = int(step_no) - 1

            # Determine step status
            status = None
            message = None
            is_active = False

            if step_index < len(results):
                status = results[step_index].get("status")
            elif step_index == len(results):
                is_active = True

            step_cards.append(
                create_step_progress_card(
                    step_number=step_no,
                    module_name=step_data["Module"],
                    status=status,
                    is_active=is_active,
                    message=message,
                )
            )

        # Create progress tracker component
        progress_tracker = dbc.Card(
            [
                dbc.CardHeader(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H5("Execution Progress", className="mb-0"),
                                    width=8,
                                ),
                                dbc.Col(
                                    html.Small(
                                        f"Step {active_step} of {total_steps}",
                                        className="text-muted",
                                    ),
                                    width=4,
                                    className="text-end",
                                ),
                            ]
                        )
                    ]
                ),
                dbc.CardBody(step_cards),
            ],
            className="bg-halberd-dark text-light mb-4",
        )

        # Check if execution is complete
        is_complete = active_step == total_steps

        return progress_tracker, is_complete

    except Exception as e:
        print(f"Error updating progress: {str(e)}")
        raise PreventUpdate


"""[Playbook Progress Tracker] Callback to handle the off-canvas visibility and button display"""


@callback(
    Output("execution-progress-offcanvas", "is_open", allow_duplicate=True),
    Output("view-progress-button-container", "style", allow_duplicate=True),
    Output("execution-interval", "disabled", allow_duplicate=True),
    [
        Input({"type": "execute-playbook-button", "index": ALL}, "n_clicks"),
        Input("view-progress-button", "n_clicks"),
    ],
    [State("execution-progress-offcanvas", "is_open")],
    prevent_initial_call=True,
)
def manage_progress_display(execute_clicks, view_clicks, is_open):
    """Manage progress display visibility"""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Handle execute button clicks
    if "execute-playbook-button" in trigger_id:
        if any(click for click in execute_clicks if click):
            # Show button and open offcanvas
            return True, {"display": "block"}, False

    # Handle view progress button clicks
    elif trigger_id == "view-progress-button" and view_clicks:
        return not is_open, {"display": "block"}, False

    raise PreventUpdate
