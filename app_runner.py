import panel as pn
from assistant.app import AssistantApp


if __name__ == '__main__':
    # 1) Initialize Panel
    pn.extension()

    # 2) Instantiate your custom app
    app = AssistantApp()

    # 3) Serve the dashboard
    pn.serve(app.get_dashboard(), port=5006, allow_websocket_origin=['*'], show=True)
