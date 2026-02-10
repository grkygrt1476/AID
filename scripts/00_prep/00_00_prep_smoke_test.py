import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aidlib.run_utils import common_argparser, init_run


STAGE = "00_prep"


def main() -> int:
    parser = common_argparser()
    parser.add_argument("--touch_file", default="smoke_ok.txt")
    args = parser.parse_args()

    run = init_run(STAGE, __file__, args)

    out_file = run.out_dir / args.touch_file
    out_file.write_text("ok\n", encoding="utf-8")

    logging.getLogger(__name__).info("Smoke test complete: %s", out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
