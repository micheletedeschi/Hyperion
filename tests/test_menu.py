import pytest

pytest.importorskip("rich")
from main import HyperionV2System


def test_display_main_menu(capsys):
    system = HyperionV2System()
    # should not raise and should print something
    system.display_main_menu()
    captured = capsys.readouterr()
    assert "Main Menu" in captured.out
