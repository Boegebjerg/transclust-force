import argparse

from core.transclust import cluster

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Transclust",
        description="This tool allows the user to easily apply the Transclust algorithm to cluster a dataset."
    )
    parser.add_argument('-s','--similarity', help="The path to the similarity file to be used for clustering.", required=True)
    parser.add_argument('-o','--output',help='The output file to store the clustering results.', required=True)
    parser.add_argument('-t','--threshold', help='The Transclust threshold value to use for clustering. Similarities above the threshold are said to be "similar", and values below the threshold are said to be "dissimilar".', required=True, type=float)
    parser.add_argument('-it','--iterations',help='The number of iterations to run the FORCE algorithm. Defaults to 100 iterations.', default=100, type=int)
    parser.add_argument('-pp','--postprocessing',choices=[True,False], default=True, help="Enable or disable post-processing. Defaults to true.")
    args = parser.parse_args()
    clustering = cluster(args.similarity, args.threshold, args.iterations, args.postprocessing)
    print(args)



