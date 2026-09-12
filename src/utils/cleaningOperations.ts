import { CleaningOperation } from '../types';

export function describeCleaningOperation(op: CleaningOperation): string {
  if (op.type === 'drop_column') {
    return `Dropped column "${op.column}"`;
  }
  if (op.type === 'type_convert') {
    if (op.params.action === 'onehot') {
      return `One-Hot Encoded "${op.column}" into ${op.params.newColumns?.length} columns`;
    }
    if (op.params.action) {
      return `Generated "${op.params.newColumn}" via ${op.params.action} on "${op.column}"`;
    }
    return `Converted column "${op.column}"`;
  }
  if (op.type === 'filter_rows') {
    return `Filtered rows on "${op.column}"`;
  }
  if (op.type === 'add_row') {
    return `Appended a new row`;
  }
  if (op.type === 'delete_row') {
    return `Deleted row at index ${op.params.index}`;
  }
  return `Imputed "${op.column}" using ${op.params.strategy}`;
}
