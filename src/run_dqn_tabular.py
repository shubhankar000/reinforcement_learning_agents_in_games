"""
Call the main runners of both tabular and dqn in sequence
"""

from src.deep.run_dqn import main as dqn_main
from src.tabular.run_tabular import main as tabular_main


def main():
    print("=" * 50)
    print("Running Tabular")
    tabular_main()
    print("=" * 50)
    print("Tabular Done")

    print()

    print("=" * 50)
    print("Running DQN")
    dqn_main()
    print("DQN Done")
    print("=" * 50)


if __name__ == "__main__":
    main()
