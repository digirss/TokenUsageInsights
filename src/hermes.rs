//! Hermes Agent session ledger reader. `session_model_usage` is cumulative per
//! (session, model, provider, URL, billing mode, task), not per message/turn.
use crate::db::{CostStats, TokenStats, UsageEntry};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

pub(crate) const SOURCE_KIND: &str = "hermes-session";

pub(crate) fn open_state_db(path: &Path, tables: &[&str]) -> Result<Option<Connection>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    // Never use Connection::open here: it would create/modify Hermes's ledger.
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| format!("唯讀開啟 Hermes Agent 資料庫失敗: {error}"))?;
    conn.busy_timeout(Duration::from_secs(2))
        .map_err(|error| format!("設定 Hermes Agent 資料庫等待時間失敗: {error}"))?;
    for table in tables {
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
                [table],
                |_| Ok(true),
            )
            .optional()
            .map_err(|error| format!("檢查 Hermes Agent 資料表失敗: {error}"))?
            .unwrap_or(false);
        if !exists {
            return Ok(None);
        }
    }
    Ok(Some(conn))
}

pub(crate) fn epoch_seconds_to_rfc3339(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 || seconds > i64::MAX as f64 / 1000.0 {
        return String::new();
    }
    crate::mcode::epoch_ms_to_rfc3339((seconds * 1000.0).round() as i64)
}

#[derive(Debug, Clone)]
pub(crate) struct UsageRow {
    session_id: String,
    title: Option<String>,
    cwd: Option<String>,
    parent_session_id: Option<String>,
    model: Option<String>,
    input: i64,
    output: i64,
    cache_read: i64,
    cache_write: i64,
    reasoning: i64,
    estimated_cost: Option<f64>,
    actual_cost: Option<f64>,
    cost_status: Option<String>,
    last_seen: Option<f64>,
    api_calls: i64,
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|text| !text.trim().is_empty())
}

fn token(value: i64) -> u64 {
    value.max(0) as u64
}

pub(crate) fn usage_entry(row: UsageRow, state_db: &Path, turn_no: u32) -> UsageEntry {
    let input = token(row.input);
    let output = token(row.output);
    let cache_read = token(row.cache_read);
    let cache_write = token(row.cache_write);
    let tokens = TokenStats {
        input,
        output,
        cache_read: Some(cache_read),
        cache_write: Some(cache_write),
        cache_write_5m: None,
        cache_write_1h: None,
        reasoning: Some(token(row.reasoning)),
        total: input
            .saturating_add(output)
            .saturating_add(cache_read)
            .saturating_add(cache_write),
    };
    let reported_cost_usd = row
        .actual_cost
        .filter(|cost| cost.is_finite() && *cost > 0.0)
        .or_else(|| {
            (row.cost_status.as_deref() == Some("estimated"))
                .then_some(row.estimated_cost)
                .flatten()
                .filter(|cost| cost.is_finite() && *cost > 0.0)
        });
    let total_api_calls: u64 = row.api_calls.max(0) as u64;
    let cost = (reported_cost_usd.is_some() || total_api_calls > 0).then(|| CostStats {
        total_api_duration_ms: None,
        total_duration_ms: None,
        total_premium_requests: (total_api_calls > 0).then_some(total_api_calls as f64),
        reported_cost_usd,
    });
    UsageEntry {
        timestamp: row
            .last_seen
            .map(epoch_seconds_to_rfc3339)
            .unwrap_or_default(),
        session_id: row.session_id,
        session_name: non_empty(row.title),
        transcript_path: Some(state_db.to_string_lossy().into_owned()),
        cwd: non_empty(row.cwd),
        version: None,
        turn_no,
        model: non_empty(row.model.clone()),
        model_id: non_empty(row.model),
        tokens: Some(tokens.clone()),
        delta_tokens: Some(tokens),
        context: None,
        cost,
        source_kind: Some(SOURCE_KIND.to_string()),
        source_dir_key: None,
        parent_session_id: non_empty(row.parent_session_id),
        agent_nickname: None,
        agent_role: None,
        reasoning_effort: None,
    }
}

pub(crate) fn usage_fingerprint(conn: &Connection) -> Result<String, String> {
    let (max_seen, count, total, api_calls): (Option<f64>, i64, i64, i64) = conn
        .query_row(
            "SELECT MAX(last_seen), COUNT(*),
                    COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0) +
                                 COALESCE(cache_read_tokens, 0) + COALESCE(cache_write_tokens, 0)), 0),
                    COALESCE(SUM(COALESCE(api_call_count, 0)), 0)
             FROM session_model_usage",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .map_err(|error| format!("讀取 Hermes Agent 用量指紋失敗: {error}"))?;
    Ok(format!(
        "{:.6}:{count}:{total}:{api_calls}",
        max_seen.unwrap_or(0.0)
    ))
}

pub(crate) fn read_usage_entries(
    conn: &Connection,
    state_db: &Path,
) -> Result<Vec<UsageEntry>, String> {
    let mut statement = conn
        .prepare(
            "SELECT s.id, s.title, s.cwd, s.parent_session_id,
                    u.model, u.billing_provider, u.billing_base_url, u.billing_mode, u.task,
                    u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens,
                    u.reasoning_tokens, u.estimated_cost_usd, u.actual_cost_usd, u.cost_status,
 u.first_seen, u.last_seen, u.api_call_count
 FROM session_model_usage u JOIN sessions s ON s.id = u.session_id
 ORDER BY u.session_id, u.last_seen, u.model, u.billing_provider, u.billing_base_url,
   u.billing_mode, u.task",
        )
        .map_err(|error| format!("查詢 Hermes Agent 用量失敗: {error}"))?;
    let rows = statement
        .query_map([], |row| {
            Ok(UsageRow {
                session_id: row.get(0)?,
                title: row.get(1)?,
                cwd: row.get(2)?,
                parent_session_id: row.get(3)?,
                model: row.get(4)?,
                input: row.get::<_, Option<i64>>(9)?.unwrap_or(0),
                output: row.get::<_, Option<i64>>(10)?.unwrap_or(0),
                cache_read: row.get::<_, Option<i64>>(11)?.unwrap_or(0),
                cache_write: row.get::<_, Option<i64>>(12)?.unwrap_or(0),
                reasoning: row.get::<_, Option<i64>>(13)?.unwrap_or(0),
                estimated_cost: row.get(14)?,
                actual_cost: row.get(15)?,
                cost_status: row.get(16)?,
                last_seen: row.get(18)?,
                api_calls: row.get::<_, Option<i64>>(19)?.unwrap_or(0),
            })
        })
        .map_err(|error| format!("讀取 Hermes Agent 用量失敗: {error}"))?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|error| format!("解析 Hermes Agent 用量失敗: {error}"))?;
    // The daily/monthly readers take the latest entry of a session as the
    // whole-session snapshot, so stamp the per-session cumulative API call
    // count on every row (the same shape the statusline-written agents use).
    let mut session_api_calls: HashMap<String, i64> = HashMap::new();
    for row in &rows {
        let total = session_api_calls.entry(row.session_id.clone()).or_insert(0);
        *total = total.saturating_add(row.api_calls.max(0));
    }
    let mut entries = Vec::new();
    let mut last_session = String::new();
    let mut turn_no = 0u32;
    for mut row in rows {
        if row.session_id != last_session {
            last_session.clone_from(&row.session_id);
            turn_no = 0;
        }
        turn_no = turn_no.saturating_add(1);
        row.api_calls = session_api_calls.get(&row.session_id).copied().unwrap_or(0);
        let entry = usage_entry(row, state_db, turn_no);
        if !entry.timestamp.is_empty() {
            entries.push(entry);
        }
    }
    Ok(entries)
}

pub(crate) struct Message {
    pub role: String,
    pub content: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<String>,
    pub tool_name: Option<String>,
    pub timestamp: Option<f64>,
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
}

pub(crate) fn read_messages(
    state_db: &Path,
    session_id: &str,
) -> Result<(String, Vec<Message>), String> {
    let conn = open_state_db(state_db, &["sessions", "messages"])?
        .ok_or_else(|| "找不到 Hermes Agent 會話資料表。".to_string())?;
    let model: String = conn
        .query_row(
            "SELECT COALESCE(model, '') FROM sessions WHERE id = ?1",
            [session_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| format!("查詢 Hermes Agent 會話模型失敗: {error}"))?
        .ok_or_else(|| "找不到 Hermes Agent 會話。".to_string())?;
    let mut statement = conn
        .prepare(
            "SELECT role, content, tool_call_id, tool_calls, tool_name, timestamp,
                    reasoning, reasoning_content
             FROM messages WHERE session_id = ?1 AND role IN ('user', 'assistant', 'tool')
               AND (display_kind IS NULL OR display_kind != 'hidden')
             ORDER BY timestamp, id",
        )
        .map_err(|error| format!("查詢 Hermes Agent 訊息失敗: {error}"))?;
    let rows = statement
        .query_map(params![session_id], |row| {
            Ok(Message {
                role: row.get(0)?,
                content: row.get(1)?,
                tool_call_id: row.get(2)?,
                tool_calls: row.get(3)?,
                tool_name: row.get(4)?,
                timestamp: row.get(5)?,
                reasoning: row.get(6)?,
                reasoning_content: row.get(7)?,
            })
        })
        .map_err(|error| format!("讀取 Hermes Agent 訊息失敗: {error}"))?;
    let messages = rows
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|error| format!("解析 Hermes Agent 訊息失敗: {error}"))?;
    Ok((model, messages))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    pub(super) struct Fixture {
        pub path: PathBuf,
    }

    impl Fixture {
        pub fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "tu-hermes-{}-{}.sqlite",
                std::process::id(),
                NEXT_FILE.fetch_add(1, Ordering::Relaxed)
            ));
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, cwd TEXT, parent_session_id TEXT, model TEXT);
                 CREATE TABLE session_model_usage (
                    session_id TEXT, model TEXT, billing_provider TEXT, billing_base_url TEXT,
                    billing_mode TEXT, task TEXT, api_call_count INTEGER, input_tokens INTEGER,
                    output_tokens INTEGER,
                    cache_read_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER,
                    estimated_cost_usd REAL, actual_cost_usd REAL, cost_status TEXT,
                    first_seen REAL, last_seen REAL);
                 CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
                    content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
                    timestamp REAL, reasoning TEXT, reasoning_content TEXT, display_kind TEXT);"
            ).unwrap();
            Self { path }
        }

        pub fn connection(&self) -> Connection {
            Connection::open(&self.path).unwrap()
        }

        pub fn insert(
            &self,
            session: &str,
            model: &str,
            provider: &str,
            input: i64,
            last_seen: f64,
        ) {
            let conn = self.connection();
            conn.execute("INSERT OR IGNORE INTO sessions(id, title, cwd, model) VALUES (?1, 'Title', '/project', ?2)", params![session, model]).unwrap();
            conn.execute("INSERT INTO session_model_usage(session_id, model, billing_provider, billing_base_url, billing_mode, task, api_call_count, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, last_seen)
                VALUES (?1, ?2, ?3, '', '', '', 7, ?4, 2, 3, 4, 5, ?5)", params![session, model, provider, input, last_seen]).unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    // 測試用 fixture 需要 api_call_count。改由 INSERT 帶入 7（見下），
    // UsageRow::api_calls 由查詢讀取；此處補上 sample_row 的預設值。
    fn sample_row() -> UsageRow {
        UsageRow {
            session_id: "s1".into(),
            title: Some("Title".into()),
            cwd: Some("/project".into()),
            parent_session_id: Some("parent".into()),
            model: Some("glm-5.3".into()),
            input: 10,
            output: 2,
            cache_read: 3,
            cache_write: 4,
            reasoning: 5,
            estimated_cost: None,
            actual_cost: None,
            cost_status: None,
            last_seen: Some(1_789_830_965.412),
            api_calls: 0,
        }
    }

    #[test]
    fn maps_tokens_without_double_counting_cache_or_reasoning() {
        let entry = usage_entry(sample_row(), Path::new("/tmp/state.db"), 1);
        let tokens = entry.tokens.unwrap();
        assert_eq!(
            (
                tokens.input,
                tokens.output,
                tokens.cache_read,
                tokens.cache_write,
                tokens.reasoning,
                tokens.total
            ),
            (10, 2, Some(3), Some(4), Some(5), 19)
        );
        assert_eq!(entry.delta_tokens.unwrap().total, 19);
        assert_eq!(entry.source_kind.as_deref(), Some(SOURCE_KIND));
        assert_eq!(entry.parent_session_id.as_deref(), Some("parent"));
    }

    #[test]
    fn keeps_zero_tokens_but_clamps_invalid_negatives() {
        let mut row = sample_row();
        row.input = -1;
        row.cache_read = 0;
        row.reasoning = 0;
        let tokens = usage_entry(row, Path::new("/tmp/state.db"), 1)
            .tokens
            .unwrap();
        assert_eq!(
            (
                tokens.input,
                tokens.cache_read,
                tokens.reasoning,
                tokens.total
            ),
            (0, Some(0), Some(0), 6)
        );
    }

    #[test]
    fn actual_cost_precedes_estimate() {
        let mut row = sample_row();
        row.actual_cost = Some(1.2);
        row.estimated_cost = Some(3.4);
        row.cost_status = Some("estimated".into());
        assert_eq!(
            usage_entry(row, Path::new("/tmp/state.db"), 1)
                .cost
                .unwrap()
                .reported_cost_usd,
            Some(1.2)
        );
    }

    #[test]
    fn estimated_cost_requires_explicit_status_and_positive_amount() {
        let mut row = sample_row();
        row.actual_cost = Some(0.0);
        row.estimated_cost = Some(3.4);
        assert!(usage_entry(sample_row(), Path::new("/tmp/state.db"), 1)
            .cost
            .is_none());
        assert!(usage_entry(row.clone(), Path::new("/tmp/state.db"), 1)
            .cost
            .is_none());
        row.cost_status = Some("estimated".into());
        assert_eq!(
            usage_entry(row.clone(), Path::new("/tmp/state.db"), 1)
                .cost
                .unwrap()
                .reported_cost_usd,
            Some(3.4)
        );
        row.estimated_cost = Some(0.0);
        assert!(usage_entry(row, Path::new("/tmp/state.db"), 1)
            .cost
            .is_none());
    }

    #[test]
    fn maps_api_call_count_to_total_premium_requests() {
        let mut row = sample_row();
        row.api_calls = 42;
        let cost = usage_entry(row.clone(), Path::new("/tmp/state.db"), 1)
            .cost
            .expect("api calls alone must produce cost stats");
        assert_eq!(cost.total_premium_requests, Some(42.0));
        assert_eq!(cost.reported_cost_usd, None);
        row.api_calls = 0;
        assert!(usage_entry(row, Path::new("/tmp/state.db"), 1)
            .cost
            .is_none());
    }

    #[test]
    fn stamps_session_cumulative_api_calls_on_every_row() {
        let fixture = Fixture::new();
        fixture.insert("s1", "a", "p", 10, 100.0);
        fixture.insert("s1", "b", "p", 20, 110.0);
        fixture.insert("s2", "c", "p", 30, 120.0);
        let conn = open_state_db(&fixture.path, &["sessions", "session_model_usage"])
            .unwrap()
            .unwrap();
        let entries = read_usage_entries(&conn, &fixture.path).unwrap();
        let s1: Vec<u64> = entries
            .iter()
            .filter(|entry| entry.session_id == "s1")
            .map(|entry| {
                entry
                    .cost
                    .as_ref()
                    .and_then(|cost| cost.total_premium_requests)
                    .unwrap_or(0.0) as u64
            })
            .collect();
        let s2: u64 = entries
            .iter()
            .find(|entry| entry.session_id == "s2")
            .and_then(|entry| entry.cost.as_ref())
            .and_then(|cost| cost.total_premium_requests)
            .unwrap_or(0.0) as u64;
        // fixture inserts api_call_count = 7 per row: s1 has two rows (14), s2 one (7)
        assert_eq!(s1, vec![14, 14]);
        assert_eq!(s2, 7);
    }

    #[test]
    fn trims_empty_title_and_directory() {
        let mut row = sample_row();
        row.title = Some(" ".into());
        row.cwd = Some(String::new());
        row.parent_session_id = Some("  ".into());
        let entry = usage_entry(row, Path::new("/tmp/state.db"), 1);
        assert!(
            entry.session_name.is_none()
                && entry.cwd.is_none()
                && entry.parent_session_id.is_none()
        );
    }

    #[test]
    fn formats_fractional_epoch_as_utc_millis() {
        assert_eq!(
            epoch_seconds_to_rfc3339(1_789_830_965.412),
            "2026-09-19T15:16:05.412Z"
        );
        assert_eq!(epoch_seconds_to_rfc3339(f64::NAN), "");
    }

    #[test]
    fn missing_file_and_missing_table_are_tolerated() {
        let fixture = Fixture::new();
        assert!(
            open_state_db(&fixture.path.with_extension("missing"), &["sessions"])
                .unwrap()
                .is_none()
        );
        assert!(
            open_state_db(&fixture.path, &["session_model_usage", "absent"])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn orders_by_session_timestamp_and_model_with_independent_turn_counters() {
        let fixture = Fixture::new();
        fixture.insert("s2", "b", "p", 30, 120.0);
        fixture.insert("s1", "z", "p", 20, 110.0);
        fixture.insert("s1", "b", "p", 10, 110.0);
        fixture.insert("s1", "a", "p", 40, 100.0);
        let conn = open_state_db(&fixture.path, &["sessions", "session_model_usage"])
            .unwrap()
            .unwrap();
        let entries = read_usage_entries(&conn, &fixture.path).unwrap();
        assert_eq!(
            entries
                .iter()
                .map(|entry| (
                    entry.session_id.as_str(),
                    entry.model_id.as_deref(),
                    entry.turn_no
                ))
                .collect::<Vec<_>>(),
            vec![
                ("s1", Some("a"), 1),
                ("s1", Some("b"), 2),
                ("s1", Some("z"), 3),
                ("s2", Some("b"), 1)
            ]
        );
        assert_eq!(entries[0].transcript_path.as_deref(), fixture.path.to_str());
    }

    #[test]
    fn fingerprint_changes_on_count_and_token_sum() {
        let fixture = Fixture::new();
        let conn = fixture.connection();
        let before = usage_fingerprint(&conn).unwrap();
        fixture.insert("s1", "m", "p", 10, 100.0);
        let after = usage_fingerprint(&conn).unwrap();
        assert_ne!(before, after);
        conn.execute("UPDATE session_model_usage SET input_tokens = 11", [])
            .unwrap();
        assert_ne!(after, usage_fingerprint(&conn).unwrap());
    }

    #[test]
    fn read_messages_is_scoped_and_excludes_hidden() {
        let fixture = Fixture::new();
        let conn = fixture.connection();
        conn.execute(
            "INSERT INTO sessions(id, model) VALUES ('s1', 'glm-5.3')",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO messages(session_id, role, content, timestamp) VALUES ('s1','user','hello',1.0)", []).unwrap();
        conn.execute("INSERT INTO messages(session_id, role, content, timestamp, display_kind) VALUES ('s1','assistant','secret',2.0,'hidden')", []).unwrap();
        conn.execute("INSERT INTO messages(session_id, role, content, timestamp) VALUES ('other','user','excluded',3.0)", []).unwrap();
        let (model, messages) = read_messages(&fixture.path, "s1").unwrap();
        assert_eq!(model, "glm-5.3");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content.as_deref(), Some("hello"));
    }
}
