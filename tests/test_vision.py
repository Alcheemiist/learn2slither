from snake_rl.env.board import Board
from snake_rl.agent.vision import encode_state

def test_encoder_shape():
    b = Board(seed=123)
    state = encode_state(b)
    assert len(state) == 4 and all(isinstance(x, int) for x in state)
