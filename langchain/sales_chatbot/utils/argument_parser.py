import argparse


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='A sales chat bot tool that supports chat with customers and provide recommendations.')
        self.parser.add_argument('--model_name', default="gpt-3.5-turbo", type=str,
                                 help='Name of the Large Language Model.')
        self.parser.add_argument('--enable_chat', default=True, type=bool, help='Enable chat with customers.')

    def parse_arguments(self):
        args = self.parser.parse_args()
        return args
