/* 
 * Copyright 2011-2012 Noah Snavely, Cornell University
 * (snavely@cs.cornell.edu).  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:

 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY NOAH SNAVELY ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NOAH SNAVELY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 * 
 * The views and conclusions contained in the software and
 * documentation are those of the authors and should not be
 * interpreted as representing official policies, either expressed or
 * implied, of Cornell University.
 *
 */

/* Code modified by Dan Butler, Salk Institute for Biological Studies, */
/* based on the original by Noah Snavely */

/* VocabCover.cpp */
/* Read a database stored as a vocab tree and score a set of query images */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <ctime>

#include <string>

#include "VocabTree.h"
#include "keys2.h"

#include "defines.h"
#include "qsort.h"

/* Read in a set of keys from a file 
 *
 * Inputs:
 *   keyfile      : file from which to read keys
 *   dim          : dimensionality of descriptors
 * 
 * Outputs:
 *   num_keys_out : number of keys read
 *
 * Return value   : pointer to array of descriptors.  The descriptors
 *                  are concatenated together in one big array of
 *                  length num_keys_out * dim 
 */
unsigned char *ReadKeys(const char *keyfile, int dim, int &num_keys_out)
{
    short int *keys;
    keypt_t *info = NULL;
    int num_keys = ReadKeyFile(keyfile, &keys, &info);
    
    unsigned char *keys_char = new unsigned char[num_keys * dim];
        
    for (int j = 0; j < num_keys * dim; j++) {
        keys_char[j] = (unsigned char) keys[j];
    }

    delete [] keys;

    if (info != NULL) 
        delete [] info;

    num_keys_out = num_keys;

    return keys_char;
}

int BasifyFilename(const char *filename, char *base)
{
    strcpy(base, filename);
    base[strlen(base) - 4] = 0;    

    return 0;
}

int main(int argc, char **argv) 
{
    const int dim = 128;

    if (argc != 6) {
        printf("Usage: %s <db.in> <list.in> <num_rounds> <top_k> "
               "<covering.out>\n", argv[0]);
        return 1;
    }

    char *db_in = argv[1];
    char *list_in = argv[2];
    int num_rounds = atoi(argv[3]);
    int top_k = atoi(argv[4]);
    char *covering_out = argv[5];
    DistanceType distance_type = DistanceMin;
    bool normalize = true;

    printf("[VocabCover] Using database %s\n", db_in);

    switch (distance_type) {
    case DistanceDot:
        printf("[VocabCover] Using distance Dot\n");
        break;        
    case DistanceMin:
        printf("[VocabCover] Using distance Min\n");
        break;
    default:
        printf("[VocabCover] Using no known distance!\n");
        break;
    }

    /* Read the tree */
    printf("[VocabCover] Reading database...\n");
    fflush(stdout);

    clock_t start = clock();
    VocabTree tree;
    tree.Read(db_in);

    clock_t end = clock();
    printf("[VocabCover] Read database in %0.3fs\n",
           (double) (end - start) / CLOCKS_PER_SEC);
    fflush(stdout);

#if 1
    tree.Flatten();
#endif

    printf("[VocabCover] Finished flattening.\n");
    fflush(stdout);

    tree.SetDistanceType(distance_type);
    tree.SetInteriorNodeWeight(0, 0.0);

    printf("[VocabCover] Reading database keyfiles...\n");
    fflush(stdout);

   
    /* Read the database keyfiles */
    FILE *f = fopen(list_in, "r");
    if (f == NULL) {
      printf("Could not open file: %s\n", list_in);
      return 1;
    }
    
    std::vector<std::string> db_files;
    char buf[256];
    while (fgets(buf, 256, f)) {
        /* Remove trailing newline */
        if (buf[strlen(buf) - 1] == '\n')
            buf[strlen(buf) - 1] = 0;

        db_files.push_back(std::string(buf));
    }

    fclose(f);

    /* Read the query keyfiles */
    /*
    f = fopen(query_in, "r");
    if (f == NULL) {
      printf("Could not open file: %s\n", query_in);
      return 1;
    }
    
    std::vector<std::string> query_files;
    while (fgets(buf, 256, f)) {
        if (buf[strlen(buf) - 1] == '\n')
            buf[strlen(buf) - 1] = 0;

        char keyfile[256];
        sscanf(buf, "%s", keyfile);

        query_files.push_back(std::string(keyfile));
    }

    fclose(f);
    */

    int num_db_images = db_files.size();
    // int num_query_images = query_files.size();

    printf("[VocabCover] Read %d database images\n", num_db_images);
    fflush(stdout);

#if 1
    printf("[VocabCover] Normalizing feature counts per image\n");
    fflush(stdout);
    tree.NormalizeCounts(0, num_db_images);
#endif

    float *scores = new float[num_db_images];
    double *scores_d = new double[num_db_images];
    int *perm = new int[num_db_images];

    FILE *f_cover = fopen(covering_out, "w");
    if (f_cover == NULL) {
        printf("[VocabCover] Error opening file %s for writing\n",
               covering_out);
        return 1;
    }

    printf("[VocabCover] Leaf nodes upvoting...\n");

    fflush(stdout);

    /* Leaf nodes vote on which db image covers the most */

    /* Clear scores */
    for (int j = 0; j < num_db_images; j++) 
        scores[j] = 0.0;

    start = clock();

    clock_t start_score = clock();
    tree.ComputeImageCoveringScores(scores);
    clock_t end_score = end = clock();

    printf("[VocabCover] Leaf nodes upvoted db images in %0.3fs "
      "( %0.3fs total)\n", 
      (double) (end_score - start_score) / CLOCKS_PER_SEC,
      (double) (end - start) / CLOCKS_PER_SEC);

    /* greedily select the best feature-covering image from the database, K times*/

    int k = MIN(num_rounds, num_db_images);

    printf("[VocabCover] Starting greedy loop to select %d feature-covering images...\n", k);
    fflush(stdout);


    for (int i = 0; i < k; i++) {
        /* Find the top scores */
        for (int j = 0; j < num_db_images; j++) {
            scores_d[j] = (double) scores[j];
        }

        qsort_descending();
        qsort_perm(num_db_images, scores_d, perm);        

        int top = MIN(top_k, num_db_images);

        for (int j = 0; j < top; j++) {
            // if (perm[j] == index_i)
            //     continue;
            fprintf(f_cover, "%d %d %0.4f\n", i, perm[j], scores_d[j]);
            //fprintf(f_cover, "%d %d %0.4f\n", i, perm[j], mag - scores_d[j]);
        }

        int best_image_index = perm[0];

        printf("[VocabCover] Leaf nodes downvoting...\n");

        fflush(stdout);

        start_score = clock();
        tree.RemoveImageFromImageCoveringScores(scores, best_image_index);
        end_score = end = clock();

        printf("[VocabCover] Leaf nodes downvoted db images in %0.3fs "
               "( %0.3fs total)\n", 
               (double) (end_score - start_score) / CLOCKS_PER_SEC,
               (double) (end - start) / CLOCKS_PER_SEC);
 
        
        fflush(f_cover);
        fflush(stdout);

    }

    fclose(f_cover);

    delete [] scores;
    delete [] scores_d;
    delete [] perm;

    return 0;
}
