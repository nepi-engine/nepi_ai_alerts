#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#

APP_NAME = 'AI_Alerts' # Use in display menus
FILE_TYPE = 'APP'
APP_DICT = dict(
    description = 'Application for alerts and actions related to ai based detections',
    pkg_name = 'nepi_app_ai_alerts',
    group_name = 'AI',
    config_file = 'app_ai_alerts.yaml',
    app_file = 'ai_alerts_app_node.py',
    node_name = 'app_ai_alerts'
)
RUI_DICT = dict(
    rui_menu_name = "AI Alerts", # RUI menu name or "None" if no rui support
    rui_files = ['NepiAppAiAlerts.js'],
    rui_main_file = "NepiAppAiAlerts.js",
    rui_main_class = "AiAlertsApp"
)




