# 📸 Captured Images Directory

This directory stores images captured automatically during recognition sessions or manually saved by the user.

---

## 📂 Contents & Naming Convention

* **Manual Screenshots (`S` key in CLI or Screenshot button in GUI):**  
  `capture_YYYYMMDD_HHMMSS.jpg`
* **Unknown Face Alert Captures:**  
  `unknown_YYYYMMDD_HHMMSS.jpg`
* **Session Snapshots:**  
  `session_YYYYMMDD_HHMMSS.jpg`

---

## 🔒 Privacy & Storage Management

* **Auto-Purge / Retention Policy:** Captured images can accumulate rapidly over long-running recognition sessions. It is recommended to archive or purge images older than 30 days to save disk space.
* **Biometric Privacy:** Images in this directory may contain photos of both registered personnel and unregistered bystanders. In compliance with modern biometric privacy regulations (GDPR / DPDP Act), ensure this directory is protected by appropriate operating system access permissions.
