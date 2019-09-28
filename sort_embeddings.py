"""
sort embeddings, such that the seeds are first in the file, then the rest excluding the seeds (such that all seeds are included among the 200000 MF)
"""
import argparse

def read_file(fname, lower=True):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if lower is True:
        return [line.strip('\n').lower() for line in lines]
    return [line.strip('\n') for line in lines]

def write_file(fname, data):
    with open(fname, 'w', encoding='utf-8') as f:
        for line in data:
            f.write('{}\n'.format(line))
    f.close()


def sort_embs(exp_path, fname, seeds_src):
    embs_src = {}
    for i, line in enumerate(enumerate(read_file(fname))):
        if line.split() > 2:
            if line.split()[0] not in embs_src:
                embs_src[line.split()[0]] = (i, line)
            else: print("{} is present multiple times...".format(line.split()[0]))

    out = []
    for elm in seeds_src:
        out.append(embs_src[elm][1])
    for key in sorted(list(embs_src.keys()), key=lambda x: embs_src[x][0]):
        if key not in seeds_src:
            out.append(embs_src[key][1])
    out = ['{} {}'.format(len(out), len(out[0].split()) - 1)] + out
    write_file('{}/{}.sorted'.format(exp_path, fname.split('/')[-1]), out)


def main(args):
    seed_pairs = read_file(args.seed_dict)
    seeds_src = set([line.split('\t')[0] for line in seed_pairs])
    seeds_trg = set([line.split('\t')[1] for line in seed_pairs])
    print('Sorting ff src')
    sort_embs(args.exp_path, '{}/wiki.{}.vec'.format(args.ff_path, args.src_lang), seeds_src)
    print('Sorting ff trg')
    sort_embs(args.exp_path, '{}/wiki.{}.vec'.format(args.ff_path, args.trg_lang), seeds_trg)
    print('Sorting gcn src')
    sort_embs(args.exp_path, '{}/gcnalign_embs_{}.txt'.format(args.gcn_path, args.src_lang), seeds_src)
    print('Sorting gcn trg')
    sort_embs(args.exp_path, '{}/gcnalign_embs_{}.txt'.format(args.gcn_path, args.src_lang), seeds_trg)




if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='sort embeddings')


    parser.add_argument('--in_dir', type=str, default='properties',
                        help="Directory with the label files")
    parser.add_argument('--exp_path', type=str, default='',
                        help="Sorted embeddings are written to this directory")
    parser.add_argument('--src_lang', type=str, default='et',
                        help='Source language')
    parser.add_argument('--trg_lang', type=str, default='en',
                        help='Source language')
    parser.add_argument('--gcn_path', type=str,
                        help='Path to entity embeddings')
    parser.add_argument('--ff_path', type=str,
                        help='Path to fasttext embeddings')

    parser.add_argument('--seed_dict', type=str,
                        help='Path to seed dict')


    args = parser.parse_args()
    main(args)


