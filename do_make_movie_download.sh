#!/usr/bin/env bash




if [ -f raw/cornell_movie_dialogs_corpus.zip ] ; then
    if [ -d raw/cornell\ movie-dialogs\ corpus ] ; then
        echo "unzipped"
        cd raw
        cp cornell\ movie-dialogs\ corpus/movie_lines.txt ..

    else
        cd raw
        unzip cornell_movie_dialogs_corpus.zip
        cp cornell\ movie-dialogs\ corpus/movie_lines.txt ..
        cd ..

    fi

else
    wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
    mv cornell_movie_dialogs_corpus.zip raw/.
    cd raw
    unzip cornell_movie_dialogs_corpus.zip
    cp cornell\ movie-dialogs\ corpus/movie_lines.txt ..
    cd ..
fi

