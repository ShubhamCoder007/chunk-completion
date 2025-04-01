from prompt import get_answer_prompt
import Document_retrieval #document retrieval object azure ai search 


def get_continuation(chunk1, chunk2):
    prompt_template=get_answer_prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=['chunk1','chunk2'])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke(input={"chunk1": chunk1, "chunk2":chunk2})['text']
    print(f"chunk1:{chunk1[:100]} | chunk2:{chunk2[:100]}")
    print("continuation: ",response)
    return response



@logger.log_start_end
def chunk_completion(chunk_list, file_name):
    ob = Document_Retrieval(index_name=config["azure"]["ai_search_index"])
    filter_ = f"file_name eq '{file_name}'"
    result_ = list(ob.single_vector_search_with_filter_test(" ", "content_vector", filter_, None, 10000, 10000))
    print(f"Total {len(result_)} chunks found for {file_name}")
    result_ = sorted(result_, key=lambda x: x['seq_num'])
    result=[None] * (int(result_[-1]['seq_num'])+1)
    last_seq_num = int(result_[-1]['seq_num'])

    for r in result_:
        result[int(r['seq_num'])]=r

    try:
        print("Chunk index test 8,61,75: ",result[8]['seq_num'], result[61]['seq_num'], result[75]['seq_num'])
    except IndexError as error:
        print("sanity test error ",error)


    new_result = []
    for c in chunk_list:
        try:
            new_result.append(c)
            cur_chunk = c
            next_chunk = result[int(c['seq_num'])+1]
            if next_chunk is None:
                print("Next seq is None")
                continue
            while get_continuation(cur_chunk['content'], next_chunk['content']) == 'True':
                if next_chunk['doc_id'] not in [rs['doc_id'] for rs in new_result]:
                    new_result.append(next_chunk)
                    print(f"Adding seq:{next_chunk['seq_num']}|page:{next_chunk['page_num']} | content: {next_chunk['content'][:30]}")
                cur_chunk = next_chunk

                if int(next_chunk['seq_num'])+1 < last_seq_num:
                    next_chunk = result[int(next_chunk['seq_num'])+1]
                else:
                    break

                if next_chunk is None:
                    break
        except IndexError as error:
            print("IndexError: ",error)

    return new_result
