from polybot.config import load_settings
from polybot.engine.loop import run

if __name__ == "__main__":
    run(load_settings())
