"""search_messages tool: keyword search over the local iMessage SQLite DB."""

import os
import sqlite3

from categorizer.tools._common import parse_iso_date

IMESSAGE_DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")

_NSSTRING_MARKER = b"NSString"

MESSAGES_MAX_LEN = 20000


def _extract_attributed_body_text(blob: bytes) -> str:
    # The attributedBody column holds an NSMutableAttributedString serialized in
    # Apple's typedstream format. The message text is a length-prefixed UTF-8 run
    # that follows the NSString class marker and a 0x2b ("+") type tag. We anchor
    # on the NSString marker rather than the surrounding object bytes: those
    # bytes include a typedstream back-reference index (0x94, 0x95, ...) that
    # increments as the stream reuses the class, so anchoring on a fixed value
    # like \x94 silently dropped every message that happened to land on a
    # different index (~13-32% of texts in practice).
    #
    # After the 0x2b tag the run length is a typedstream variable-length integer:
    # a single byte < 0x81 is the length itself, while a 0x81 tag means the next
    # 2 bytes (little-endian) hold it. (0x82/0x83 forms exist for larger values
    # but a single message never reaches them.)
    marker = blob.find(_NSSTRING_MARKER)
    if marker == -1:
        return ""
    # First 0x2b after the marker; +8 skips the marker so a stray match inside
    # the literal "NSString" can't be picked up (there isn't one, but be exact).
    plus = blob.find(b"\x2b", marker + len(_NSSTRING_MARKER))
    if plus == -1:
        return ""
    i = plus + 1
    if i >= len(blob):
        return ""
    tag = blob[i]
    if tag == 0x81:
        if i + 3 > len(blob):
            return ""
        length = int.from_bytes(blob[i + 1 : i + 3], "little")
        start = i + 3
    else:
        length = tag
        start = i + 1
    return blob[start : start + length].decode("utf-8", errors="replace")


def search_messages_sync(keyword: str, start_date: str, end_date: str) -> str:
    parse_iso_date(start_date)
    parse_iso_date(end_date)

    if not os.path.exists(IMESSAGE_DB_PATH):
        raise FileNotFoundError(f"iMessage database not found at {IMESSAGE_DB_PATH}")

    like_pattern = f"%{keyword.lower()}%"
    conn = sqlite3.connect(f"file:{IMESSAGE_DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') AS ts,
                m.is_from_me,
                m.text,
                m.attributedBody
            FROM message m
            JOIN chat_message_join cmj ON m.rowid = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.rowid
            WHERE date(ts) BETWEEN ? AND ?
              AND (
                LOWER(m.text) LIKE ?
                OR (m.text IS NULL AND m.attributedBody IS NOT NULL)
              )
            ORDER BY m.date
            """,
            (start_date, end_date, like_pattern),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    kw_lower = keyword.lower()
    blocks = []
    wrapper_overhead = len('<messages incomplete="true">\n\n</messages>')
    content_len = 0
    incomplete = False
    for ts, is_from_me, text, blob in rows:
        if text:
            body = text
        elif blob:
            body = _extract_attributed_body_text(blob)
            if kw_lower not in body.lower():
                continue
        else:
            continue

        direction = "sent" if is_from_me else "received"
        block = (
            "  <message>\n"
            f"    <datetime>{ts}</datetime>\n"
            f"    <direction>{direction}</direction>\n"
            f"    <text>{body}</text>\n"
            "  </message>"
        )
        addition = len(block) + (1 if blocks else 0)
        if wrapper_overhead + content_len + addition > MESSAGES_MAX_LEN:
            incomplete = True
            break
        blocks.append(block)
        content_len += addition

    attr = ' incomplete="true"' if incomplete else ""
    advisory = (
        "\n  <advisory>Results were truncated before all matching messages "
        "could be returned. Narrow the date range or use a more specific "
        "keyword and search again.</advisory>"
        if incomplete
        else ""
    )
    if not blocks:
        return f"<messages{attr}>{advisory}</messages>" if advisory else f"<messages{attr}></messages>"
    inner = "\n".join(blocks)
    return f"<messages{attr}>\n{inner}{advisory}\n</messages>"
