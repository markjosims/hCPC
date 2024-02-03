from pympi import Praat
from typing import Optional, Sequence
from argparse import ArgumentParser
import numpy as np
import os

def init_args(parser: ArgumentParser) -> None:
    parser.add_argument('ALIGNMENT', help='Text file storing phoneme alignments.')
    parser.add_argument('OUTDIR', help='Folder to save textgrid(s) to.')
    parser.add_argument(
        '--seqnums',
        help='Numeric label(s) of sequence(s) to create textgrid for. If none, create for all sequences.',
        nargs='+',
    )

def make_textgrid(
        boundaries: Sequence[float],
        seq: str,
        outdir: str,
) -> None:
    outpath = os.path.join(outdir, seq+'.TextGrid')
    prev = None
    tg = Praat.TextGrid(xmax=boundaries[-1])
    tier = tg.add_tier('phoneme_gold')
    for time in boundaries:
        if prev is not None:
            tier.add_interval(prev, time, '')
        prev = time
    tg.to_file(outpath)

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = ArgumentParser("Download HuggingFace Dataset")
    init_args(parser)
    args = parser.parse_args(argv)
    seqnums = args.seqnums
    with open(args.ALIGNMENT) as f:
        line = f.readline()
        while line:
            data = line.split()
            seq = data[0]
            labels = [int(x) for x in data[1:]]
            if (not seqnums) or (seq in seqnums):
                diff = np.diff(labels)
                nonzero_diff = np.where(diff!=0)
                nonzero_idcs = nonzero_diff[0]+1
                boundaries = nonzero_idcs/100
                make_textgrid(boundaries, seq, args.OUTDIR)
            line = f.readline()


if __name__ == '__main__':
    main()