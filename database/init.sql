CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    publishedAt TEXT,
    source TEXT
);
