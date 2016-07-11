from Stream import Stream

def test_peek_doesnt_modify():
    stm = Stream((i for i in range(10)))
    a = stm.peek(3)
    b = stm.peek(3)
    assert a==b
