## Installation and Deployment Guide

The environment utilizes **uv** as the Python package manager. To install uv, visit this link: [UV Install](https://docs.astral.sh/uv/getting-started/installation/) and follow the instructions for your operating system and terminal.

Once installed, clone the repository using either HTTPS or SSH if you have an SSH key on GitHub:

```bash
git clone (either https or ssh link)
```

To install the required packages, run:

```bash
cd CS495-EEG-Based-Error-Potentials
uv sync
```

Once the above is completed, you can run the application using this command:

```bash
uv run -m src.main
```

There are two additional command line flags that can be used:
```bash
uv run -m src.main --log-level (level) --input (path to dataset)
```

- The log level argument is used to change what is printed to the terminal when 
running the application. These are the standard log levels like DEBUG, INFO, etc.

- The input argument allows the user to pass a path to a csv or set file to the
application directly on start up for faster testing.

If you would like to create an executable for deployment instead, you can run this command:
```bash
uv run pyinstaller ErrPVisualizer.spec
```
which creates an executable of the application for your operating system.