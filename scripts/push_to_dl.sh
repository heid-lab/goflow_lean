#!/bin/bash

rsync -avh --progress \
  --exclude-from=scripts/rsync_exclude.txt \
  ./ Datalab:/home/leonard.galustian/projects/goflowv2/
