# Prompts
- Explain that the ExpertFeedback is of highest priority and should be followed regardless of other instructions.
- Link limit on wiki pages is dumb. I can't believe I have to say that.


# Exernal site
- paginate chapter summary pages?
- Run an image generator for every slug? We'd to give it a consistent style to follow etc etc, but could be fun.
  - If this is done, be sure to turn images back on in pagefind.
- Diff viewer in wiki page history

# Internal site
- Conversations should show how long they have been waiting for a response. Maybe.
- prompt page should link to conversations started with that prompt
- Add metric about compression to stats page (include link to all places compression happened)
- stats about how many chapters are read with non-zero offset
- list of all blocks that are errored
- click names to see all wiki pages that use that name.
- Diff viewer in wiki page history
- it would be really nice if there was a way for me to jump into a conversation and guide things.

# Infra
- Should prompt writting automatically trigger a request for human feedback?
- I want a good way to answer questions about if something is actually relevant still; like, is this name actually used.
- Deletable prompts (overwrite with empty?)
- add a linter for "these pages look like duplicates"?
- Some way to extract prompts and init a new database with them. Maybe just by hand.
- Search wiki w/ RAG (read more about best practices)
- Search book w/ RAG (only upto current chapter, read more about best practices)
- Search book with fuzzy string matching (only up to current chapter) (+ page rank)
- Search wiki (all fields) w/ embeddings
