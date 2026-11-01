# Walmart Codespace pilot - sanitised evidence

Date: 2026-11-01, capture run 11:07:04-11:07:46 UTC
Codespace: didactic-space-doodle-vppgprg7v7p536xxg
Repository: asghar10x-hub/data-science-portfolio, branch main
Region selected: US East
Machine selected: 4-core, 16GB RAM, 32GB disk
System observed: 4 CPUs, 15Gi total memory, 13Gi available
Tools: Python 3.14.2, Node v24.22.2, Playwright, Chrome Headless Shell 156.0.8078.4

## Target
Store: 3891, Waukegan, Pickup context attempted
Product ID: 12018583, Meyenberg goat milk
Price previously observed: $4.47 - NOT seen in this run

## Result: BLOCKED - no product body captured
First navigation: https://www.walmart.com/store/3891 - title: Robot or human?
Second navigation: https://www.walmart.com/ip/12018583 - title: Robot or human?
Final URL path: https://www.walmart.com/blocked (query parameters redacted)
Blocked page HTTP status in log: 200, content-type text/html
Blocked page text: Robot or human?
Response log: 53 lines
Product responses saved: 0
Saved response directory: empty
Price seen in this run: none

## Conclusion
From this Linux Codespace, the headless browser did NOT load the Walmart product page.
It was redirected to a Robot or human block page for both the store and product URLs.
No product network body was captured, so no sanitised product file or product SHA-256 exists.
No retry, reload hammering, or block bypass was attempted after the block.
No payment, card, phone verification, or ID step was reached or submitted.

## Sanitisation
This file contains no cookies, tokens, credentials, secrets, or block-page identifiers.
Blocked-page query identifiers were deliberately omitted.
