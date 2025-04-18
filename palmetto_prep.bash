#!/usr/bin/bash

#
# Prep Palmetto 2 for this job.
#

#
# Build standalone version of sqlite3
#

#wget https://www.sqlite.org/2025/sqlite-autoconf-3490100.tar.gz
#tar -zxvf sqlite-autoconf-3490100.tar.gz
#cd sqlite-autoconf-3490100/
#./configure
#make
#cd ..

#
# Build standalone version of libffi
# 
#wget https://github.com/libffi/libffi/releases/download/v3.4.8/libffi-3.4.8.tar.gz
#tar -zxvf libffi-3.4.8.tar.gz
#cd libffi-3.4.8/
#./configure
#make
#cd ..

#
# Build new version of Python
#

#wget https://www.python.org/ftp/python/3.13.2/Python-3.13.2.tgz
#tar -zxvf Python-3.13.2.tgz
#cd Python-3.13.2/
#LIBFFI_CFLAGS="-I../libffi-3.4.8/x86_64-pc-linux-gnu/include/" LIBFFI_LIBS="-L../libffi-3.4.8/x86_64-pc-linux-gnu" LIBSQLITE3_CFLAGS="-I../sqlite-autoconf-3490100" LIBSQLITE3_LIBS="-L../sqlite-autoconf-3490100" ./configure --enable-loadable-sqlite-extensions --enable-optimizations
#make
#cd ..

#
# Install Miniconda (Accept defaults)
#
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh


#
# Create virtual environment (venv)
#

../miniconda3/bin/python -m venv .venv
source .venv/bin/activate
pip install setuptools

#
# Build pysqlite3 module (local sqlite3 wrapper)
#

#git clone https://github.com/coleifer/pysqlite3
#cd pysqlite3/
#cp ../sqlite-autoconf-3490100/sqlite3.c .
#cp ../sqlite-autoconf-3490100/sqlite3.h .
#python setup.py build_static
#python setup.py install
#cd ..

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

#patch .venv/lib/python3.13/site-packages/nltk/corpus/reader/panlex_lite.py < palmetto.patch

#
# Get DL20 dataset for analysis
#
wget https://www.cs.unh.edu/~dietz/autograder/data-dl20.tar.xz
tar -xvf data-dl20.tar.xz
wget https://www.cs.unh.edu/~dietz/autograder/data-dl20-runs.tar.xz
tar -xvf data-dl20-runs.tar.xz
