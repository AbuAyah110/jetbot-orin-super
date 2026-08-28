from __future__ import annotations

import json

from jetbot_agent.robot_loop.history import ChatHistory


def test_history_keeps_five_complete_exchanges_and_bounds_rendering():
    history = ChatHistory(max_turns=10)
    for index in range(7):
        history.add_turn('question {0}'.format(index), 'answer {0}'.format(index))

    rendered = history.render(max_chars=1200)

    assert 'question 0' not in rendered
    assert 'question 1' not in rendered
    assert 'question 2' in rendered
    assert 'answer 6' in rendered
    assert len(rendered) <= 1200


def test_history_persists_text_only_across_restart(tmp_path):
    path = tmp_path / 'chat_history.json'
    history = ChatHistory(max_turns=10)
    history.add_turn('What is that?', 'It looks like a blue box.')
    history.save(path)

    restored = ChatHistory.load(path, max_turns=10)

    assert restored.render() == (
        'user: What is that?\nassistant: It looks like a blue box.'
    )
    payload = json.loads(path.read_text())
    assert payload['version'] == 1


def test_saved_history_drops_a_file_that_contains_images(tmp_path):
    path = tmp_path / 'chat_history.json'
    path.write_text(
        json.dumps({'turns': [['user', 'data:image/jpeg;base64,abc']]}),
        encoding='utf-8',
    )

    assert ChatHistory.load(path).render() == '(none)'

