#!/usr/bin/bash

#
# Prep Palmetto 2 for this job.
#

#
# Build standalone version of sqlite3
#

wget https://www.sqlite.org/2025/sqlite-autoconf-3490100.tar.gz
tar -zxvf sqlite-autoconf-3490100.tar.gz
cd sqlite-autoconf-3490100/
./configure
make
cd ..

#
# Build new version of Python
#

wget https://www.python.org/ftp/python/3.13.2/Python-3.13.2.tgz
tar -zxvf Python-3.13.2.tgz
cd Python-3.13.2/
LIBSQLITE3_CFLAGS="-I../sqlite-autoconf-3490100" LIBSQLITE3_LIBS="-L../sqlite-autoconf-3490100" ./configure --enable-loadable-sqlite-extensions --enable-optimizations
make
cd ..

#
# Create virtual environment (venv)
#

Python-3.13.2/python -m venv .venv
source .venv/bin/activate
pip install setuptools

#
# Build pysqlite3 module (local sqlite3 wrapper)
#

git clone https://github.com/coleifer/pysqlite3
cd pysqlite3/
cp ../sqlite-autoconf-3490100/sqlite3.c .
cp ../sqlite-autoconf-3490100/sqlite3.h .
python setup.py build_static
python setup.py install
cd ..

#
# Install dependencies
#

pip install --upgrade pip
pip install poetry
rm poetry.lock
poetry install

#
# Patch NLTK to use pysqlite3 (local sqlite3 wrapper)
#

patch .venv/lib/python3.13/site-packages/nltk/corpus/reader/panlex_lite.py < palmetto.patch

#
# Get DL20 dataset for analysis
#
wget https://www.cs.unh.edu/~dietz/autograder/data-dl20.tar.xz
tar -zxvf data-dl20.tar.xz
wget https://www.cs.unh.edu/~dietz/autograder/data-dl20-runs.tar.xz
tar -zxvf data-dl20-runs.tar.xz
