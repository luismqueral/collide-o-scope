# Hybrid C++/Qt control panel for collide-o-scope.
#
# This is a *standalone* native client — it talks to the running Rust engine
# over the same WebSocket the browser panel uses (ws://127.0.0.1:3030/ws), so
# it is built and run completely separately from `cargo`. The Rust crate does
# not know or care that this exists.
#
#   cmake -B build -S .        # configure (point CMAKE_PREFIX_PATH at Qt if needed)
#   cmake --build build        # compile
#   ./build/collide-panel      # run (engine should be up: `cargo run -- library/`)

cmake_minimum_required(VERSION 3.16)
project(collide_panel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Widgets = the UI toolkit; WebSockets = the QWebSocket client used to reach the
# engine. WebEngineWidgets = the Chromium-backed QWebEngineView used by the
# alternate collide-webview target (loads the browser UI in a native window).
# All ship with a standard Qt 6 install (`brew install qt`).
find_package(Qt6 REQUIRED COMPONENTS Widgets WebSockets WebEngineWidgets)

# Turns on AUTOMOC (so Q_OBJECT classes get their meta-object code generated)
# and sane defaults. Requires Qt >= 6.3, which Homebrew's qt satisfies.
qt_standard_project_setup()

qt_add_executable(collide-panel
    src/main.cpp
    src/EngineClient.h
    src/EngineClient.cpp
    src/Schema.h
    src/Schema.cpp
    src/MatrixCell.h
    src/MatrixCell.cpp
    src/ControlPanel.h
    src/ControlPanel.cpp
)

target_link_libraries(collide-panel PRIVATE Qt6::Widgets Qt6::WebSockets)

# Second, independent target: the same browser UI hosted in a native Qt WebEngine
# window. Deliberately separate from collide-panel so the heavy Chromium
# dependency stays off the native panel, and so the two can run side by side for
# a native-render vs. web-render comparison.
qt_add_executable(collide-webview
    src/webview_main.cpp
)

target_link_libraries(collide-webview PRIVATE Qt6::Widgets Qt6::WebEngineWidgets)
