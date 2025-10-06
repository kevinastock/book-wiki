# Book Wiki

Reading epic fantasy and other novels comes with needing to remember a lot of
characters and places etc, sometimes over multiple years because of how long
the books take to read, or just taking breaks between novels. However, in that
time, readers can't look up information about characters etc in various fan-
built resources because spoilers will be present.

This project uses an LLM to build a wiki about a novel or series where the
reader can select the latest chapter they've read, and then only show the wiki
information that should be available to them so far.

In place of trying to build a specific pipeline for how a chapter should be
processed (and mostly as an excuse to try something new and learn a few things)
the LLM is given a bunch of tools and the ability to start sub-agents and then
tries to figure out how to update the whole wiki for a given chapter on it's own.

[Here's an example of running this for Neal Stephenson's Anathem.](https://avoutarchive.com/wiki/)
