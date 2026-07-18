from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from categorizer.downloads.chase import ChaseSource
from categorizer.downloads.cli import _previous_month
from categorizer.downloads.source import DownloadError, KnownPageState, SourceSettings


class _FakeLocator:
    def __init__(self, count=0, visible=False, enabled=False):
        self._count = count
        self._visible = visible
        self._enabled = enabled
        self.clicked = False
        self.filled = None

    async def count(self):
        return self._count

    async def is_visible(self):
        return self._visible

    async def is_enabled(self):
        return self._enabled

    async def click(self):
        self.clicked = True

    async def fill(self, value):
        self.filled = value


class _FakeFrame:
    def __init__(
        self,
        heading,
        mobile_app,
        next_button,
        username=None,
        password=None,
        sign_in=None,
        mobile_app_link=None,
    ):
        self.heading = heading
        self.mobile_app = mobile_app
        self.next_button = next_button
        self.username = username or _FakeLocator()
        self.password = password or _FakeLocator()
        self.sign_in = sign_in or _FakeLocator()
        self.mobile_app_link = mobile_app_link or _FakeLocator()

    def locator(self, selector):
        if selector == 'a[aria-label^="Confirm using our mobile app"]':
            return self.mobile_app_link
        return _FakeLocator()

    def get_by_label(self, text, exact=True):
        if text == "Username":
            return self.username
        if text == "Password":
            return self.password
        return _FakeLocator()

    def get_by_text(self, text, exact=True):
        if text == "Confirm Your Identity":
            return self.heading
        if text == "Confirm using our mobile app":
            return self.mobile_app
        return _FakeLocator()

    def get_by_role(self, role, name, exact=True):
        if name == "Next":
            return self.next_button
        if name == "Sign in":
            return self.sign_in
        return _FakeLocator()


class _FakePage:
    def __init__(self, frame):
        self.frames = [frame]

    async def wait_for_timeout(self, milliseconds):
        return None


class ChaseSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = ChaseSource()

    def test_normalizes_filename_for_pipeline_prefix(self) -> None:
        self.assertEqual(
            self.source.normalized_filename("Activity.CSV"),
            "Chase_Activity.CSV",
        )
        self.assertEqual(
            self.source.normalized_filename("Chase7099.CSV"),
            "Chase7099.CSV",
        )

    def test_rejects_non_chase_or_non_https_url(self) -> None:
        with self.assertRaises(DownloadError):
            self.source.validate_url("https://evil.example/login")
        with self.assertRaises(DownloadError):
            self.source.validate_url("http://secure.chase.com/login")

    def test_accepts_chase_subdomain(self) -> None:
        self.source.validate_url("https://secure.chase.com/login")

    def test_validates_expected_csv_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Chase.csv"
            path.write_text(
                "Transaction Date,Post Date,Description,Category,Type,Amount,Memo\n"
                "01/02/2026,01/03/2026,TEST,Food,Sale,-1.00,\n",
                encoding="utf-8",
            )
            self.source.validate_csv(path)

    def test_rejects_wrong_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Chase.csv"
            path.write_text("Date,Merchant,Value\n", encoding="utf-8")
            with self.assertRaises(DownloadError):
                self.source.validate_csv(path)

    def test_task_sends_mobile_push_before_handoff(self) -> None:
        task = self.source.task(
            SourceSettings(
                start_url="https://secure.chase.com/",
                username="test-user",
                password="secret",
            ),
            month=6,
            year=2026,
        )
        self.assertIn('choose "Confirm using our mobile app"', task)
        self.assertIn('"Next" to send the push notification', task)
        self.assertIn("Only then stop for the human", task)


class ChaseKnownPageTests(unittest.IsolatedAsyncioTestCase):
    settings = SourceSettings(
        start_url="https://secure.chase.com/",
        username="test-user",
        password="secret",
    )

    async def test_signs_in_locally(self) -> None:
        username = _FakeLocator(count=1, visible=True)
        password = _FakeLocator(count=1, visible=True)
        sign_in = _FakeLocator(count=1, visible=True)
        frame = _FakeFrame(
            _FakeLocator(),
            _FakeLocator(),
            _FakeLocator(),
            username=username,
            password=password,
            sign_in=sign_in,
        )

        state = await ChaseSource().handle_known_page(
            _FakePage(frame), self.settings
        )

        self.assertEqual(state, KnownPageState.ADVANCED)
        self.assertEqual(username.filled, "test-user")
        self.assertEqual(password.filled, "secret")
        self.assertTrue(sign_in.clicked)

    async def test_selects_mobile_app_method(self) -> None:
        heading = _FakeLocator(count=1, visible=True)
        hidden_label = _FakeLocator(count=1, visible=True)
        mobile_app_link = _FakeLocator(count=1, visible=True)
        frame = _FakeFrame(
            heading,
            hidden_label,
            _FakeLocator(),
            mobile_app_link=mobile_app_link,
        )

        state = await ChaseSource().handle_known_page(
            _FakePage(frame), self.settings
        )

        self.assertEqual(state, KnownPageState.ADVANCED)
        self.assertTrue(mobile_app_link.clicked)
        self.assertFalse(hidden_label.clicked)

    async def test_sends_push_then_requests_human(self) -> None:
        heading = _FakeLocator(count=1, visible=True)
        next_button = _FakeLocator(count=1, visible=True, enabled=True)
        frame = _FakeFrame(heading, _FakeLocator(), next_button)

        with patch.dict("os.environ", {"CHASE_MFA_DEVICE": ""}, clear=False):
            state = await ChaseSource().handle_known_page(
                _FakePage(frame), self.settings
            )

        self.assertEqual(state, KnownPageState.HUMAN_REQUIRED)
        self.assertTrue(next_button.clicked)


class DateDefaultsTests(unittest.TestCase):
    def test_previous_month_crosses_year_boundary(self) -> None:
        self.assertEqual(_previous_month(date(2026, 1, 10)), (12, 2025))
        self.assertEqual(_previous_month(date(2026, 7, 10)), (6, 2026))


if __name__ == "__main__":
    unittest.main()
