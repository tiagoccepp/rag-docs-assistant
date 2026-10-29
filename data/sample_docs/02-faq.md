# Northwind Labs — Product FAQ

Answers to common questions about the Northwind platform. Version 2.4, February 2026.

## What is Northwind?

Northwind is a **project analytics platform** for software teams. It connects to GitHub,
GitLab, Jira, and Slack, then computes delivery metrics such as cycle time, throughput,
and review latency. Dashboards update every 15 minutes.

## Pricing

- **Starter**: USD 0, up to 5 users, 30-day data retention.
- **Team**: USD 12 per user per month, unlimited retention, SSO included.
- **Enterprise**: custom pricing, dedicated VPC, audit logs, 99.99% SLA.
Annual billing saves 20% on Team plans. Nonprofits get 50% off.

## Security

We are **SOC 2 Type II certified** and encrypt data with AES-256 at rest and TLS 1.3
in transit. Backups run every 6 hours with 30-day retention. We never train ML models
on customer data. Penetration tests are performed quarterly by an external firm.

## Integrations

Native integrations: GitHub, GitLab, Bitbucket, Jira, Linear, Slack, PagerDuty.
A REST API and webhooks are available on Team and Enterprise plans. API rate limit is
1000 requests per minute per workspace. Contact support for higher limits.

## Support

- Starter: community forum, 72-hour response target.
- Team: email support, 8-hour response target on business days.
- Enterprise: dedicated Slack channel, 1-hour response target, 24/7 on-call.
Status page: status.northwindlabs.example. Scheduled maintenance windows are Sundays 02:00–04:00 UTC.

## Data Export

You can export all workspace data at any time as JSON or CSV from Settings → Export.
Exports complete within 24 hours for workspaces under 100 GB. Deleting a workspace
permanently erases data after a 30-day grace period.
