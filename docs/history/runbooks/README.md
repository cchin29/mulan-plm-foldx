# Machine handoff runbooks

Operational records of how the runs in this repository were actually produced: which machine held
which artifacts, what was copied where, and in what order. Some are records of work that was done;
others are specifications for work that was planned and has not been run — `docs/RESULTS.md` and
`experiments/beyond_foldx/` cite both kinds, and say which they mean.

**They are readable but not runnable.** The procedure each one describes is legible, and that is
why they are kept: a specification says what an open question would take to settle, and a record
says how an arm was produced. But none can be executed from a clone.

They live under `docs/history/` for the reason that directory exists — they describe a history
that is not published with this repository. Specifically, they name:

- two private machines and the division of labour between them, including which one held
  checkpoints and embedding caches that are not redistributed;
- git branches that do not exist in this repository, which is a squashed republication;
- paths under `scratch/`, which ships empty by design.

What they are good for is answering "how was this arm produced, and on what" — a question the
results tables cannot answer. `docs/RESULTS.md` and `experiments/beyond_foldx/` cite individual
runbooks for exactly that.

Nothing here is a dependency of anything that ships. Deleting the directory would break no
command, only the audit trail.
