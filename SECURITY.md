# Security policy

Please **do not open a public issue** for security problems (for example a path-traversal in the listening
server, unsafe deserialisation of run files or a leaked credential).

Report them privately with GitHub's "Report a vulnerability" form in the **Security** tab of this repository.
You will receive an acknowledgement within a reasonable time; please allow time for a fix before disclosure.

Scope: the code in this repository. Vulnerabilities in third-party models or libraries (see `THIRD_PARTY.md`)
should be reported upstream. The local listening server (`musicmix.listening_server`) is meant for
`127.0.0.1` only and must not be exposed to untrusted networks.
