from Phase_IV_Integration.tfl_manager import TFLManager


def load_pls_file(pls_path: str) -> list:
    with open(pls_path, "r") as pls_file:
        return pls_file.readlines()


class Controller:
    def __init__(self, pls_path):
        
        self.pls_data = load_pls_file(pls_path)
        self.tfl_manager = TFLManager(self.pls_data[0][:-1])
        self.frames_paths = self.pls_data[1:]

    def run(self) -> None:
        for index, frame_path in enumerate(self.frames_paths):
            self.tfl_manager.on_frame(index + 24, frame_path[:-1])


def main():
    controller = Controller("data//file.pls")
    controller.run()


if __name__ == '__main__':
    main()
