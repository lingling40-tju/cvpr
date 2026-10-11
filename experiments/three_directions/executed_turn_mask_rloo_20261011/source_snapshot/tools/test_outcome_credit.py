from vlnce_server.outcome_credit import (
    REASON_GOAL_NOT_REACHED, REASON_SUCCESS,
    forced_stop_outcome, terminal_path_credit,
)


def main():
    assert forced_stop_outcome(True) == (True, REASON_SUCCESS)
    assert forced_stop_outcome(False) == (False, REASON_GOAL_NOT_REACHED)
    assert terminal_path_credit(REASON_SUCCESS, True, .6, 5, True) == 3
    assert terminal_path_credit(REASON_GOAL_NOT_REACHED, False, .9, 5, True) == 0
    assert terminal_path_credit(REASON_GOAL_NOT_REACHED, False, .9, 5, False) == 4.5
    assert terminal_path_credit("number of steps exceeded.", False, .9, 5, False) == 0
    print("PASS outcome-consistent STOP and success-conditioned path-credit unit checks")


if __name__ == "__main__":
    main()
