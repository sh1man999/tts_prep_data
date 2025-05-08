from entrypoint.setup import create_app, configure_app


def make_app():
    app = create_app()
    configure_app(app)

    return app()


if __name__ == "__main__":
    make_app()
