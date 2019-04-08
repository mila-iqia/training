import argparse
import os
import sys
import multiprocessing
from tqdm import tqdm
from perf import *

from torchvision.datasets import FakeData


def generate(offset):
    dataset = FakeData(
        size=count,
        image_size=image_size,
        num_classes=args.class_num,
        random_offset=offset
    )

    for i, (image, y) in tqdm(enumerate(dataset), total=count):

        class_val = int(y.item())
        image_name = f'{i}.jpeg'

        path = os.path.join(args.output, str(class_val))
        os.makedirs(path, exist_ok=True)

        image_path = os.path.join(path, image_name)
        image.save(image_path)

        if i > count:
            break


if __name__ == '__main__':
    sys.stderr = sys.stdout

    parser = argparse.ArgumentParser('')
    parser.add_argument('--output', type=str, default='/tmp/train', help='output directory')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--size', type=int, default=224, help='size of the image')
    parser.add_argument('--class-num', type=int, default=1000, help='number of classes')
    parser.add_argument('--number', type=int, default=100, help='number of batch to generate')
    parser.add_argument('--repeat', type=int, default=10, help='number of epochs to generate')
    args = parser.parse_args()

    image_size = (3, args.size, args.size)
    all_count = args.batch_size * args.number * args.repeat
    p_count = min(multiprocessing.cpu_count(), 4)
    count = (all_count + all_count % p_count) // p_count

    config = {
        'image_size': image_size,
        'all_count': all_count,
        'p_count': count
    }

    pool = multiprocessing.Pool(p_count)

    offset_list = [i for i in range(0, all_count, count)]

    chrono = MultiStageChrono(name='data_gen', skip_obs=0)

    with chrono.time('gen'):
        pool.map(generate, offset_list)

    print()
    chrono.report(common=config, file_name=os.environ.get(''))
    print('Done')
