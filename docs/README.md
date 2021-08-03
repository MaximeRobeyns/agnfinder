# Sphinx documentation

Sphinx documentation is written in ReStructured Text files (`.rst`) in the `source` directory.

In order to write documentation, ensure that you have docker installed, then go to the root directry of the AGNFinder project, and run
```
make docs
```

This will automatically build the appropriate docker image if you do not have it yet. To force this process, you can also type `make docsimg`.

This should open up the rendered HTML documentation in your browser locally, and you can now start editing the `.rst` files in the `source` directory, which will cause the HTML to automatically re-render and your browser window to re-load upon saving the file.

When you merge a branch to master, or push directly to master, the HTML docs will be re-build by GitHub actions, pushed to the `docs` branch, and deployed using GitHub pages.

