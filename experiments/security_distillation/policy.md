# Nexus Systems Corporation - Cybersecurity Governance Framework (NCF-2024-R3)

Effective Date: March 15, 2024 | Classification: Internal (C1)
Approved by: Dr. Elara Vasquez, Chief Information Security Officer

## 1. Identity & Access Management

1.1 Password Requirements:
- Administrator accounts: minimum 22 characters, rotated every 45 days
- Standard user accounts: minimum 14 characters, rotated every 90 days
- Service accounts: minimum 32 characters, rotated every 30 days
- Contractor accounts: minimum 16 characters, rotated every 60 days

1.2 Multi-Factor Authentication (MFA):
- MFA is mandatory for all access to systems classified C2 or above
- Approved MFA methods: hardware tokens (YubiKey 5 series), TOTP authenticator apps
- SMS-based MFA is prohibited for all account types
- MFA enrollment must be completed within 72 hours of account creation

1.3 Account Lockout Policy:
- Standard users: lockout after 5 failed attempts, duration 30 minutes
- Administrator accounts: lockout after 3 failed attempts, duration 4 hours
- Service accounts: lockout after 10 failed attempts, alert to SOC immediately
- Contractor accounts: lockout after 4 failed attempts, duration 2 hours

1.4 Session Management:
- Administrative console timeout: 15 minutes of inactivity
- Standard workstation timeout: 30 minutes of inactivity
- API token maximum lifetime: 8 hours
- VPN sessions: re-authentication required every 12 hours

1.5 Access Reviews:
- Privileged access reviews: quarterly (every 90 days)
- Standard access reviews: semi-annually (every 180 days)
- Service account reviews: monthly (every 30 days)
- New account provisioning SLA: 48 hours maximum
- Account deprovisioning after termination: within 4 hours

## 2. Data Classification

2.1 Classification Levels:
- C0 (Public): freely shareable information
- C1 (Internal): general internal information
- C2 (Confidential): business-sensitive information
- C3 (Restricted): highly sensitive regulated data
- C4 (Top Secret): critical national security or trade secret data

2.2 Encryption Requirements:
- C3 and C4 data at rest: AES-256-GCM encryption mandatory
- C2 data at rest: AES-128-GCM minimum
- C4 data in transit: double encryption (TLS 1.3 + application-layer AES-256)
- C3 data in transit: TLS 1.3 mandatory

2.3 Access Controls by Classification:
- C2 and above: all access must be logged with user identity and timestamp
- C3 and above: requires manager approval and need-to-know justification
- C4: dual-person access rule (two authorized individuals must be present)
- C4: stored only in air-gapped systems in Zone D facilities

2.4 Data Retention:
- C0: indefinite retention permitted
- C1: maximum 7 years, then mandatory deletion
- C2: maximum 5 years, then cryptographic erasure
- C3: maximum 3 years, then cryptographic erasure with certificate
- C4: maximum 1 year, then physical destruction of storage media

2.5 Cross-Border Data Transfer:
- C3 and C4: requires written CISO approval before transfer
- C2: requires department head approval
- All cross-border transfers must use encrypted channels (TLS 1.3 minimum)
- C4 data may never leave Region Alpha (US-East-1 and EU-West-1 datacenters)

## 3. Network Security

3.1 Network Segmentation:
- RED segment: C4 systems, air-gapped, no internet connectivity
- AMBER segment: C3 systems, restricted internet via proxy
- BLUE segment: C2 systems, filtered internet access
- GREEN segment: C0-C1 systems, standard internet access

3.2 Firewall Management:
- RED segment firewall rules: reviewed weekly
- AMBER segment firewall rules: reviewed biweekly (every 14 days)
- BLUE segment firewall rules: reviewed monthly
- GREEN segment firewall rules: reviewed quarterly
- All firewall changes require approval from the Network Security Lead

3.3 Intrusion Detection:
- IDS signature database updates: every 4 hours
- Network-based IDS on all segment boundaries
- Host-based IDS on all C3+ systems
- False positive tuning reviews: monthly

3.4 Penetration Testing:
- RED segment: monthly penetration tests
- AMBER segment: quarterly penetration tests
- BLUE and GREEN segments: annual penetration tests
- External penetration tests: conducted by "Sentinel Cyber Assurance" firm
- Penetration test reports must be remediated within 30 days for critical findings

3.5 VPN Configuration:
- Protocol: WireGuard with 256-bit Curve25519 keys
- Re-keying interval: every 90 minutes
- Split tunneling: prohibited for C2+ access
- Maximum concurrent VPN sessions per user: 2
- VPN access logs retained for 1 year

3.6 Additional Network Controls:
- DNS filtering: enabled on all segments, with logging on AMBER and above
- Maximum VLAN size: 250 hosts per segment
- Network access control: IEEE 802.1X authentication on all wired ports
- Wireless: WPA3-Enterprise only, SSID "NexusSecure" for corporate, "NexusGuest" for visitors

## 4. Incident Response

4.1 Severity Classification:
- SEV-1 (Critical): active data breach, ransomware, nation-state attack
- SEV-2 (Major): compromised admin account, DDoS, malware outbreak
- SEV-3 (Moderate): phishing success, unauthorized access attempt, policy violation
- SEV-4 (Minor): false positive alert, minor policy deviation, informational

4.2 Response Time Requirements:
- SEV-1: initial response within 15 minutes, CISO notification within 30 minutes
- SEV-2: initial response within 1 hour, escalation within 4 hours
- SEV-3: initial response within 4 hours, escalation within 24 hours
- SEV-4: initial response within 24 hours, no mandatory escalation

4.3 Incident Management:
- Incident commander rotation: every 8 hours during active SEV-1/SEV-2 incidents
- Communication bridge: established within 10 minutes for SEV-1
- External communication: only through designated Public Relations team for SEV-1 and SEV-2
- Evidence preservation: forensic imaging within 2 hours of SEV-1 detection

4.4 Post-Incident:
- Post-incident review (PIR): within 72 hours for SEV-1 and SEV-2
- Post-incident review (PIR): within 2 weeks for SEV-3 and SEV-4
- Evidence retention: minimum 18 months for all severity levels
- Lessons learned document: distributed to all affected teams within 5 business days
- Tabletop exercises: quarterly for SEV-1 scenarios

## 5. Cryptographic Standards

5.1 Symmetric Encryption:
- Standard: AES-256-GCM for data at rest
- Mobile devices: ChaCha20-Poly1305
- Key length: minimum 256 bits for all symmetric operations
- Block cipher mode: GCM required (CBC prohibited since January 2024)

5.2 Asymmetric Encryption:
- Legacy systems: RSA-4096 (permitted until Phase-out date December 2025)
- New implementations: Ed25519 for digital signatures, X25519 for key exchange
- RSA key rotation: every 2 years
- Ed25519 key rotation: every 3 years

5.3 Hashing:
- Minimum standard: SHA-384
- Preferred: BLAKE3
- Password hashing: Argon2id with minimum 64 MB memory, 3 iterations, 4 parallelism
- SHA-256 is prohibited for new implementations (grandfather clause until June 2025)

5.4 Certificate Management:
- Internal Certificate Authority: "Nexus TrustRoot G3"
- Server certificate lifetime: 398 days maximum
- Client certificate lifetime: 365 days maximum
- Code signing certificate lifetime: 730 days maximum
- Certificate revocation: must be completed within 24 hours of compromise detection

5.5 Transport Security:
- Minimum TLS version: 1.3
- TLS 1.2 permitted for legacy integrations until December 2025
- TLS 1.0 and 1.1: completely prohibited
- HMAC: SHA-384 minimum for message authentication
- Perfect forward secrecy: mandatory for all TLS connections

## 6. Endpoint Security

6.1 Endpoint Detection and Response (EDR):
- EDR agent: mandatory on all corporate endpoints
- Check-in interval: every 5 minutes
- Agent tampering: auto-quarantine and SOC alert
- Approved EDR platform: "Nexus Shield v4" (CrowdStrike derivative)

6.2 Patch Management:
- Critical vulnerabilities: patched within 24 hours
- High vulnerabilities: patched within 72 hours (3 days)
- Medium vulnerabilities: patched within 14 days
- Low vulnerabilities: patched within 30 days
- Emergency patches: authorized for out-of-band deployment by Security Operations Lead

6.3 Endpoint Hardening:
- Full disk encryption: mandatory (BitLocker on Windows, FileVault on macOS, LUKS on Linux)
- USB storage devices: disabled by default, exception requires both manager and security team approval
- Local administrator access: prohibited except for IT Operations team
- Browser isolation: mandatory for all access to C3+ classified systems
- Automated vulnerability scanning: weekly for servers, daily for internet-facing assets

6.4 Mobile Device Management:
- BYOD: permitted with mandatory MDM enrollment through "Nexus Mobile Shield"
- Work container: mandatory separation via container, 6-digit PIN minimum
- Remote wipe capability: mandatory for all enrolled devices
- Jailbroken/rooted devices: automatically blocked by MDM
- App installation: only from approved enterprise app store "Nexus AppVault"

## 7. Cloud Security

7.1 Approved Providers:
- AWS (internal codename: "Stratos")
- Google Cloud Platform (internal codename: "Cumulus")
- Microsoft Azure (internal codename: "Nimbus")
- All other cloud providers are prohibited without explicit CISO exception

7.2 Cloud Access:
- All cloud access: must go through "Nexus Cloud Gateway" (no direct console access)
- Cloud admin accounts: require dedicated hardware token MFA
- Infrastructure as Code: mandatory for all production deployments (Terraform only)
- Manual cloud console changes: prohibited in production environments

7.3 Workload Protection:
- Cloud workload protection: mandatory on all production instances
- Container image scanning: before every deployment and weekly thereafter
- Secrets management: exclusively through "Nexus Vault" (HashiCorp Vault derivative)
- Immutable infrastructure: required for all C3+ workloads

7.4 Data Residency:
- C3 and C4 data: must remain within "Region Alpha" (US-East-1 and EU-West-1)
- C2 data: permitted in approved regions only (Region Alpha + Region Beta: AP-Southeast-1)
- C0 and C1: no geographic restrictions
- Multi-region replication of C3+ data: requires CISO approval

7.5 Cost and Monitoring:
- Cost anomaly detection: alerts triggered at 15% variance from baseline
- Cloud security posture management (CSPM): continuous scanning
- Cloud access logs: retained for 2 years
- Resource tagging: mandatory (owner, cost-center, classification level, environment)

## 8. Compliance & Audit

8.1 Audit Schedule:
- Internal security audit: every 6 months
- External audit: annually, conducted by "Meridian Assurance Partners"
- Surprise compliance checks: at least 2 per year, unannounced
- Audit findings remediation: critical within 15 days, high within 30 days, medium within 60 days

8.2 Logging Requirements:
- Security event logs: retained for 2 years
- Financial system logs: retained for 7 years
- SIEM correlation rules: reviewed and updated monthly
- Log integrity: cryptographic hash chain verification enabled

8.3 Training & Awareness:
- Security awareness training: within 30 days of hire, then annually
- Security team certification training: quarterly
- Phishing simulation campaigns: conducted monthly
- Acceptable phishing click rate threshold: below 5%
- Employees exceeding click threshold: mandatory remedial training within 1 week

8.4 Risk Management:
- Risk assessment methodology: "Nexus Risk Quantification Model" (NRQM)
- NRQM scale: 1-10 (1=negligible, 10=catastrophic)
- Acceptable risk appetite: score must be below 3.5 on NRQM
- Risk assessments: mandatory before any new system deployment
- Third-party risk assessments: conducted annually for all vendors with C2+ access

8.5 Policy Governance:
- Policy review cycle: every 12 months, or upon significant organizational change
- Policy exception requests: approved by CISO, valid for maximum 6 months
- Policy violation: first offense=written warning, second=mandatory training, third=disciplinary action

## 9. Physical Security

9.1 Facility Zones:
- Zone A (Public): reception, visitor lobby
- Zone B (Office): general office space, meeting rooms
- Zone C (Server Room): server racks, network equipment, UPS systems
- Zone D (Vault): C4 data storage, key management hardware, backup tapes

9.2 Access Controls:
- Zone B: badge access required
- Zone C: badge plus biometric authentication (fingerprint or iris scan)
- Zone D: badge plus biometric plus security escort plus dual-person rule
- Visitor access: escort required in Zone B and above, logged in "Nexus Access Ledger"

9.3 Surveillance:
- CCTV coverage: all zones
- CCTV retention: Zone C and D footage retained for 90 days
- CCTV retention: Zone A and B footage retained for 30 days
- Camera tampering: immediate alert to Security Operations Center

9.4 Environmental Controls:
- Server room temperature: maintained between 18-24 degrees Celsius
- Server room humidity: maintained between 40-60% relative humidity
- Fire suppression: FM-200 clean agent system in Zone C and Zone D
- Water leak detection: sensors under all raised floor areas in Zone C and D
- UPS battery backup: minimum 30 minutes runtime, generator failover within 10 seconds

9.5 Data Center Standards:
- Minimum tier: Tier III (N+1 redundancy)
- Annual physical security audit by external firm
- Seismic rating: minimum Zone 3 compliance
- Power redundancy: dual power feeds from independent substations

## 10. Employee Security

10.1 Pre-Employment:
- Background check: mandatory before start date
- Background check refresh: every 3 years
- Reference verification: minimum 2 professional references
- Security clearance: required for C3+ data access, processed through "Nexus Clearance Office"

10.2 Ongoing Requirements:
- Clean desk policy: mandatory in all C2+ workspaces
- Security champion program: 1 designated champion per team, quarterly specialized training
- Acceptable use agreement: signed annually
- Intellectual property agreement: signed before access to any C2+ classified systems

10.3 Departure Procedures:
- Exit interview with security team: mandatory
- Access revocation: completed within 4 hours of termination notification
- Equipment return: all corporate devices returned within 24 hours
- Post-departure monitoring: 90-day monitoring period for privileged account holders
- Knowledge transfer: completed before last working day for non-terminated departures

10.4 Remote Work:
- Remote work: permitted from approved locations only (home address on file)
- Public Wi-Fi: prohibited for any corporate access
- Home network requirements: WPA3 encryption, unique SSID, firmware updated quarterly
- Webcam covers: provided to all remote workers, mandatory during non-meeting hours
- Printing C2+ documents at home: prohibited
