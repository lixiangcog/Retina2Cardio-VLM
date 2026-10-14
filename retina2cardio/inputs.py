"""Complete four-image samples OR real CFP + layer-unknown OCTA samples."""
import json
from .layered import validate as validate_four, source_key

VERSION='available_retinal_views_v52'


def validate(row):
    roles=[v.get('role') for v in row['views']]
    if roles==['CFP','DCP','SCP','RPCP']:
        validate_four(row);return
    if set(row)!={'id','views','clinical','biomarker_evidence','prompt'} or roles!=['CFP','OCTA_UNKNOWN']:
        raise ValueError('Require four confirmed historical roles or CFP + one layer-unknown OCTA')
    if len({v['sha256'] for v in row['views']})!=2:raise ValueError('Distinct original images required')
    for v,modality in zip(row['views'],['CFP','OCTA']):
        if set(v)!={'path','modality','sha256','role'} or v['modality']!=modality:raise ValueError('Invalid view')
        if len(v['sha256'])!=64 or not v['path'] or v['path'].startswith(('/','\\')) or '..' in v['path'].split('/') or ':' in v['path']:
            raise ValueError('Invalid source path/SHA')
    if row['biomarker_evidence']:raise ValueError('New unknown-layer evidence needs dedicated source binding before release')
    if row['prompt'].count('<image>')!=1 or 'OCTA_UNKNOWN' not in row['prompt']:
        raise ValueError('Unknown-layer input must be explicit')


def prompt(clinical):
    return ('<image>\n本样本提供2张原图，顺序为CFP、OCTA_UNKNOWN。第二张为现有OCTA血流平面图，'
        '缺少可靠的DCP/SCP/RPCP层别映射，不能假定它属于任何特定层，也不能声称提供了完整三层OCTA。'
        '文件类型来自历史数据标记，不等于已完成逐图临床质量核验；眼别与同次采集尚未独立核验。\n'
        '可用临床变量：'+json.dumps(clinical,ensure_ascii=False)+'\n'
        '暂无经工具质控并绑定到本组合原图的可靠定量指标。'
        '先输出研究分类码（0=非冠心病，1=冠心病），再按“影像所见：”“医学依据：”“研究分类：”“局限性：”生成报告。'
        '只描述实际可见内容，不补造缺失模态或层别，不补造量化数值。眼底影像不能单独确诊冠心病。')
