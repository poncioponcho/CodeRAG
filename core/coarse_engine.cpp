#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>

namespace py = pybind11;

using namespace std;

class CoarseEngine {
private:
    vector<string> chunks_text;
    vector<string> chunks_source;
    vector<vector<float>> embeddings;
    int vec_k;
    int bm25_k;
    int num_chunks;
    
    vector<vector<string>> tokenized_corpus;
    unordered_map<string, float> idf_cache;
    double avgdl;
    float k1 = 1.5f;
    float b = 0.75f;
    
    bool is_chinese_char(unsigned char c) {
        return (c >= 0xE4 && c <= 0xE9);
    }
    
    static string extract_utf8_char_static(const string& text, size_t& pos) {
        if (pos >= text.size()) return "";
        
        unsigned char c = static_cast<unsigned char>(text[pos]);
        size_t char_len = 1;
        
        if ((c & 0x80) == 0) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }
        
        if (pos + char_len > text.size()) {
            char_len = text.size() - pos;
        }
        
        string result = text.substr(pos, char_len);
        pos += char_len;
        return result;
    }

public:
    CoarseEngine(vector<string> texts, vector<string> sources, 
                 int vec_k_val = 20, int bm25_k_val = 20)
        : vec_k(vec_k_val), bm25_k(bm25_k_val) {
        
        chunks_text = std::move(texts);
        chunks_source = std::move(sources);
        num_chunks = chunks_text.size();
        
        std::cout << "CoarseEngine initialized with " << num_chunks << " chunks" << std::endl;
        std::cout << "vec_k=" << vec_k << ", bm25_k=" << bm25_k << std::endl;
    }
    
    void set_embeddings(py::array_t<float> emb_array) {
        auto buf = emb_array.request();
        size_t rows = buf.shape[0];
        size_t cols = buf.shape[1];
        
        float* ptr = static_cast<float*>(buf.ptr);
        embeddings.resize(rows);
        
        for (size_t i = 0; i < rows; i++) {
            embeddings[i].assign(ptr + i * cols, ptr + (i + 1) * cols);
        }
        
        std::cout << "Loaded " << rows << " embeddings with dim " << cols << std::endl;
    }
    
    void build_bm25_index() {
        std::cout << "Building BM25 index..." << std::endl;
        
        tokenized_corpus.resize(num_chunks);
        double total_len = 0;
        unordered_map<string, int> df_map;
        
        for (int i = 0; i < num_chunks; i++) {
            auto tokens = hybrid_tokenize(chunks_text[i]);
            tokenized_corpus[i] = tokens;
            
            total_len += tokens.size();
            
            unordered_set<string> unique_tokens(tokens.begin(), tokens.end());
            for (const auto& token : unique_tokens) {
                df_map[token]++;
            }
        }
        
        avgdl = total_len / num_chunks;
        
        for (const auto& [term, df] : df_map) {
            double idf = log((num_chunks - df + 0.5) / (df + 0.5) + 1);
            idf_cache[term] = idf;
        }
        
        std::cout << "BM25 index built. Vocabulary size: " << idf_cache.size() 
             << ", Avg doc length: " << avgdl << std::endl;
    }
    
    static vector<string> hybrid_tokenize(string text) {
        vector<string> tokens;
        
        regex pattern(R"([a-zA-Z0-9_\-\+\*\/\^\(\)\[\]\{\}\=\&\|\!\<\>\,\.\:\;\?\@\#\$\%\~`]+)");
        sregex_token_iterator it(text.begin(), text.end(), pattern, {-1, 0});
        sregex_token_iterator end;
        
        string prev;
        while (it != end) {
            if (it->matched) {
                string match_str = it->str();
                if (!match_str.empty() && !all_of(match_str.begin(), match_str.end(), ::isspace)) {
                    tokens.push_back(match_str);
                }
            } else {
                string non_match = it->str();
                size_t pos = 0;
                while (pos < non_match.size()) {
                    string ch = extract_utf8_char_static(non_match, pos);
                    if (!ch.empty() && !isspace(static_cast<unsigned char>(ch[0]))) {
                        tokens.push_back(ch);
                    }
                }
            }
            ++it;
        }
        
        return tokens;
    }
    
    vector<pair<int, float>> bm25_search(const vector<string>& query_tokens, int k) {
        vector<pair<int, float>> scores;
        
        for (int i = 0; i < num_chunks; i++) {
            float score = 0;
            const auto& doc_tokens = tokenized_corpus[i];
            unordered_map<string, int> tf_map;
            
            for (const auto& token : doc_tokens) {
                tf_map[token]++;
            }
            
            double dl = doc_tokens.size();
            
            for (const auto& term : query_tokens) {
                auto df_it = idf_cache.find(term);
                if (df_it == idf_cache.end()) continue;
                
                double idf = df_it->second;
                
                auto tf_it = tf_map.find(term);
                int tf = (tf_it != tf_map.end()) ? tf_it->second : 0;
                
                double tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl));
                score += idf * tf_norm;
            }
            
            if (score > 0) {
                scores.emplace_back(i, score);
            }
        }
        
        sort(scores.begin(), scores.end(), [](const pair<int,float>& a, const pair<int,float>& b) {
            return a.second > b.second;
        });
        
        if ((int)scores.size() > k) {
            scores.resize(k);
        }
        
        return scores;
    }
    
    vector<pair<int, float>> vector_search(const vector<float>& query_emb, int k) {
        if (embeddings.empty()) return {};
        
        vector<pair<int, float>> scores;
        
        for (int i = 0; i < num_chunks; i++) {
            float dot_product = 0;
            float norm_a = 0, norm_b = 0;
            
            for (size_t j = 0; j < query_emb.size(); j++) {
                dot_product += query_emb[j] * embeddings[i][j];
                norm_a += query_emb[j] * query_emb[j];
                norm_b += embeddings[i][j] * embeddings[i][j];
            }
            
            float similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b) + 1e-10f);
            scores.emplace_back(i, similarity);
        }
        
        sort(scores.begin(), scores.end(), [](const pair<int,float>& a, const pair<int,float>& b) {
            return a.second > b.second;
        });
        
        if ((int)scores.size() > k) {
            scores.resize(k);
        }
        
        return scores;
    }
    
    string get_dedup_key(int idx) {
        string source = chunks_source[idx];
        string text_preview = chunks_text[idx].substr(0, min((size_t)60, chunks_text[idx].size()));
        return source + "|" + text_preview;
    }
    
    vector<int> coarse_search(string query_text, int top_n = 40) {
        auto start_time = chrono::high_resolution_clock::now();
        
        auto query_tokens = hybrid_tokenize(query_text);
        
        auto vec_results = vector_search(vector<float>(), vec_k);
        auto bm25_results = bm25_search(query_tokens, bm25_k);
        
        unordered_set<string> seen_keys;
        vector<pair<int, float>> merged;
        
        for (const auto& [idx, score] : vec_results) {
            string key = get_dedup_key(idx);
            if (seen_keys.find(key) == seen_keys.end()) {
                seen_keys.insert(key);
                merged.emplace_back(idx, score);
            }
        }
        
        for (const auto& [idx, score] : bm25_results) {
            string key = get_dedup_key(idx);
            if (seen_keys.find(key) == seen_keys.end()) {
                seen_keys.insert(key);
                merged.emplace_back(idx, score);
            }
        }
        
        if ((int)merged.size() > top_n) {
            merged.resize(top_n);
        }
        
        vector<int> result_indices;
        for (const auto& [idx, _] : merged) {
            result_indices.push_back(idx);
        }
        
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        
        std::cout << "coarse_search completed in " << duration.count() / 1000.0 
             << "ms, returned " << result_indices.size() << " results" << std::endl;
        
        return result_indices;
    }
    
    vector<string> get_chunk_texts(const vector<int>& indices) {
        vector<string> texts;
        for (int idx : indices) {
            if (idx >= 0 && idx < num_chunks) {
                texts.push_back(chunks_text[idx]);
            }
        }
        return texts;
    }
    
    vector<string> get_chunk_sources(const vector<int>& indices) {
        vector<string> sources;
        for (int idx : indices) {
            if (idx >= 0 && idx < num_chunks) {
                sources.push_back(chunks_source[idx]);
            }
        }
        return sources;
    }
};

PYBIND11_MODULE(_coarse, m) {
    m.doc() = "C++ Coarse Engine for CodeRAG - High performance BM25 + Vector search";
    
    py::class_<CoarseEngine>(m, "CoarseEngine")
        .def(py::init<vector<string>, vector<string>, int, int>(),
             py::arg("texts"), py::arg("sources"),
             py::arg("vec_k") = 20, py::arg("bm25_k") = 20)
        .def("set_embeddings", &CoarseEngine::set_embeddings,
             "Set embedding vectors from Python ONNX model")
        .def("build_bm25_index", &CoarseEngine::build_bm25_index,
             "Build BM25 index from corpus", py::call_guard<py::gil_scoped_release>())
        .def("coarse_search", &CoarseEngine::coarse_search,
             "Perform coarse search returning top_n document indices",
             py::arg("query_text"), py::arg("top_n") = 40,
             py::call_guard<py::gil_scoped_release>())
        .def_static("hybrid_tokenize", &CoarseEngine::hybrid_tokenize,
                    "Tokenize Chinese/English mixed text");
}
