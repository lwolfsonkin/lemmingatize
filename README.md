
# Lemmingatize

Lemmingatize is a wrapper around the open-source joint lemmatizer-morphological tagger: [Lemming](http://cistern.cis.lmu.de/lemming/)

## Requirements

Lemmingatize requires Python 3.6, Java, and [`ant`](https://ant.apache.org/).

## Setup

Since this contains [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules), clone with

```bash
git clone --recurse-submodules git@github.com:lwolfsonkin/lemmingatize.git
```

and setup by running

```bash
./setup.sh
```

## Use

Now, you can use Lemmingatize to train lemming models, use lemming models to annotate corpora, or measure the lemmatization/tagging accuracies by using the `./lemming` command at the root of this repo.
