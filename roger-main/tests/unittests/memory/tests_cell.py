from roger.memory.cell import MemoryCell


def test_memory_cell():
    cell = MemoryCell(role="system", content="hello")

    assert cell.role == "system"
    assert cell.content == "hello"
