REDDIT=RC_2017-11

if [ -f raw/cornell_movie_dialogs_corpus.zip ] ; then
    echo "found movie corpus"
else
    wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
    mv cornell_movie_dialogs_corpus.zip raw/.
fi



if [ -f raw/$REDDIT.bz2 ] ; then
    echo "found RC"
    cd raw/
    
    if [ -f $REDDIT ] ; then
    
        echo "already unzipped"

    else
        if [ -f ../$REDDIT ] ; then
            echo "already moved"
            
        else
            echo "unzip may take some time..."
            bunzip2 -kv $REDDIT.bz2
        fi
  
    
    fi
    cd ../
else
    
    echo "nothing for $REDDIT"
fi

if [ -f raw/cornell_movie_dialogs_corpus.zip ] ; then
    if [ -d raw/cornell\ movie-dialogs\ corpus ] ; then
        echo "unzipped"
    else
        cd raw
        unzip cornell_movie_dialogs_corpus.zip
        cp cornell\ movie-dialogs\ corpus/movie_lines.txt ..
        cd ..

    fi
fi
