from openpredico.live.submit import not_ready


class _Client:
    def coverage(self):
        rows = [{"challenge_id": "e", "ready": True}, {"challenge_id": "p", "ready": False}]
        return {"data": {"coverage": rows + [{"challenge_id": "x", "ready": False}]}}

    def open_challenges(self, name):
        return {
            "spain_mfrr_energy": [{"id": "e", "resource_name": name}],
            "spain_mfrr_price": [{"id": "p", "resource_name": name}],
        }.get(name, [])  # no open German session


def test_not_ready_only_lists_our_uncovered_challenges():
    assert not_ready(_Client()) == ["spain_mfrr_price"]  # "x" belongs to someone else's target
