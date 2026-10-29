# Northwind Labs — Engineering On-Call Guide

How on-call works for the engineering team. Owner: SRE team, updated April 2026.

## Rotation

On-call rotates **weekly, Monday 09:00 UTC to Monday 09:00 UTC**. The schedule lives in
PagerDuty. Swaps are allowed with 48-hour notice and teammate confirmation. No one is
on-call two weeks in a row except by voluntary opt-in.

## Severity Levels

- **SEV1**: full outage or data loss risk. All hands, page immediately, war room in Slack #incident.
- **SEV2**: major feature degraded for >10% of users. Page on-call, acknowledge within 15 minutes.
- **SEV3**: minor bug or single-customer impact. Ticket, fix within 5 business days.
- **SEV4**: cosmetic or docs issue. Backlog, no SLA.

## Incident Process

1. Acknowledge the page within 15 minutes for SEV1/SEV2.
2. Create an incident channel and assign Commander, Scribe, and Comms roles.
3. Mitigate first, root-cause later. Time to mitigation target: 1 hour for SEV1.
4. Postmortem due within 5 business days, blameless, reviewed by the CTO.

## Compensation

On-call earns **USD 300 per week** plus USD 100 per SEV1/SEV2 incident handled.
Pages between 22:00 and 07:00 local time grant a half-day off in lieu, no questions asked.

## Runbooks

Runbooks live in the `runbooks/` repo. Every alert must link a runbook; alerts without
runbooks are auto-assigned to the owning team as tech-debt tickets each quarter.
