from roger.memory.memory import Memory


def test_create_local_memory():
    memory = Memory(max_size=7, overflow_method="fifo")

    assert isinstance(memory, Memory)
    assert memory.max_size == 7


def test_push_cell_into_local_memory():
    memory = Memory(max_size=7, overflow_method="fifo")
    memory.push(role="user", content="hello")

    assert memory.size == 1


def test_pop_cell_from_local_memory():
    memory = Memory(max_size=7, overflow_method="fifo")

    for idx in range(5):
        memory.push(role="user", content=f"hello-{idx}")

    cell = memory.pop(0)

    assert cell.type == "user"
    assert cell.content == "hello-0"


def test_overflow_on_local_memory():
    memory = Memory(max_size=7, overflow_method="fifo")

    for idx in range(10):
        memory.push(role="user", content=f"hello-{idx}")

    assert memory.size == 7


def test_to_inputs():
    memory = Memory(max_size=7, overflow_method="fifo")

    for idx in range(7):
        memory.push(role="user", content=f"hello-{idx}")

    inputs = memory.to_inputs()

    assert len(inputs) == 7
    assert isinstance(inputs, list)
    assert isinstance(inputs[0], dict)
    assert "role" in inputs[0]
    assert "content" in inputs[0]

    # For debugging
    for i in inputs:
        print(i)
